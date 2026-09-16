"""独立命令入口，避免包初始化先导入 prompting 后再重复执行该模块。"""

from .prompting import main

if __name__ == "__main__":
    main()
