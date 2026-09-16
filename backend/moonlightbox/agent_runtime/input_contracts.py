"""按明确的参数责任投影工具输入，不改变持久化模型或推断业务内容。

omit 由工具作者声明。投影是真实的 Pydantic 类型（不是删 JSON Schema），
因此嵌套 Schema 与输入校验一致；重建后仍须通过原领域模型的所有校验。
"""

from copy import deepcopy
from functools import reduce
from operator import or_
from types import UnionType
from typing import Annotated, Union, get_args, get_origin

from pydantic import BaseModel, create_model


def input_model(model, *, omit):
    cache = {}

    def annotation(value):
        if isinstance(value, type) and issubclass(value, BaseModel):
            return project(value)
        origin, args = get_origin(value), get_args(value)
        if origin is Annotated:
            return Annotated[annotation(args[0]), *args[1:]]
        if origin in (Union, UnionType):
            return reduce(or_, (annotation(item) for item in args))
        if origin in (list, tuple, dict, set):
            return origin[tuple(annotation(item) for item in args)]
        return value

    def project(cls):
        if cls in cache:
            return cache[cls]
        fields = {
            name: (annotation(field.annotation), deepcopy(field))
            for name, field in cls.model_fields.items()
            if not omit(cls, name, field)
        }
        # 不继承原类，否则被移除的字段仍会继承回来；领域验证在恢复绑定值后执行。
        projected = create_model(
            cls.__name__ + "Input", __config__={**cls.model_config, "extra": "forbid"}, **fields
        )
        cache[cls] = projected
        return projected

    return project(model)
