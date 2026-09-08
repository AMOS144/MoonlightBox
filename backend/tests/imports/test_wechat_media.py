import hashlib
import sqlite3
import struct
from pathlib import Path

import pytest
import zstandard
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def test_wechat_media_resolves_database_path_and_md5(tmp_path: Path) -> None:
    from moonlightbox.imports.wechat_media import WechatMediaExtractor

    database_path = tmp_path / "message.db"
    media_root = tmp_path / "media"
    media_root.mkdir()
    (media_root / "emoji.gif").write_bytes(b"gif")
    (media_root / "abcdef.png").write_bytes(b"png")
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE resource_x (message_id TEXT, md5 TEXT, local_path TEXT)")
    connection.executemany(
        "INSERT INTO resource_x VALUES (?, ?, ?)",
        [
            ("m1", None, "emoji.gif"),
            ("m2", "abcdef", None),
        ],
    )
    connection.commit()
    connection.close()

    extractor = WechatMediaExtractor()
    records = extractor.records(database_path)
    resolved = extractor.resolve(records, [media_root])

    assert [item.record.message_id for item in resolved] == ["m1", "m2"]


def test_wechat_media_rejects_unsafe_database_path(tmp_path: Path) -> None:
    from moonlightbox.imports.wechat_media import WechatMediaExtractor

    database_path = tmp_path / "message.db"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE resource_x (message_id TEXT, md5 TEXT, local_path TEXT)")
    connection.execute("INSERT INTO resource_x VALUES ('m1', NULL, '../secret')")
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="路径不安全"):
        WechatMediaExtractor().records(database_path)


def test_wechat_media_reads_zstd_sticker_and_avatar(tmp_path: Path) -> None:
    from moonlightbox.imports.wechat_media import WechatMediaExtractor

    message_directory = tmp_path / "message"
    message_directory.mkdir()
    contact_username = "wxid_target"
    table = "Msg_" + hashlib.md5(contact_username.encode()).hexdigest()
    connection = sqlite3.connect(message_directory / "message_0.db")
    connection.execute("CREATE TABLE Name2Id (user_name TEXT)")
    connection.execute("INSERT INTO Name2Id VALUES (?)", (contact_username,))
    connection.execute(
        f'CREATE TABLE "{table}" '
        "(local_id INTEGER, create_time INTEGER, real_sender_id INTEGER, "
        "local_type INTEGER, message_content BLOB)"
    )
    xml = (
        b'<msg><emoji md5="0123456789abcdef0123456789abcdef" '
        b'cdnurl="https://emoji.qpic.cn/example"/></msg>'
    )
    connection.execute(
        f'INSERT INTO "{table}" VALUES (?, ?, ?, ?, ?)',
        (7, 123, 1, 47, zstandard.ZstdCompressor().compress(xml)),
    )
    connection.commit()
    connection.close()

    head_image_database = tmp_path / "head_image.db"
    connection = sqlite3.connect(head_image_database)
    connection.execute("CREATE TABLE head_image (username TEXT, image_buffer BLOB)")
    connection.execute(
        "INSERT INTO head_image VALUES (?, ?)",
        (contact_username, b"\xff\xd8\xffavatar"),
    )
    connection.commit()
    connection.close()

    extractor = WechatMediaExtractor()
    stickers = extractor.contact_stickers(
        message_directory,
        contact_username,
    )

    assert stickers[0].sender_username == contact_username
    assert stickers[0].md5 == "0123456789abcdef0123456789abcdef"
    assert extractor.avatar_bytes(head_image_database, contact_username) == b"\xff\xd8\xffavatar"


def test_wechat_media_extracts_store_sticker_package(tmp_path: Path) -> None:
    from moonlightbox.imports.wechat_media import WechatMediaExtractor

    content = b"GIF89a-sticker"
    sticker_md5 = hashlib.md5(content).hexdigest()
    package_id = "wechat-sticker-package"
    package_hash = hashlib.md5(package_id.encode()).hexdigest()
    package_path = (
        tmp_path / "business" / "emoticon" / "PersistStore" / package_hash[:2] / package_hash
    )
    package_path.parent.mkdir(parents=True)
    package_path.write_bytes(b"prefix" + content)

    database_path = tmp_path / "emoticon.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE kStoreEmoticonFilesTable "
        "(package_id_ TEXT, md5_ TEXT, emoticon_offset_ INTEGER, "
        "emoticon_size_ INTEGER)"
    )
    connection.execute(
        "INSERT INTO kStoreEmoticonFilesTable VALUES (?, ?, ?, ?)",
        (package_id, sticker_md5, 6, len(content)),
    )
    connection.commit()
    connection.close()

    assert (
        WechatMediaExtractor().store_sticker_bytes(
            database_path,
            tmp_path,
            sticker_md5,
        )
        == content
    )


def test_wechat_v4_image_dat_is_decoded_to_browser_image() -> None:
    from moonlightbox.imports.wechat_media import decode_wechat_image_dat

    image = b"\xff\xd8\xff" + b"jpeg-body-" * 8 + b"\xff\xd9"
    aes_key = b"0123456789abcdef"
    xor_key = 0x0F
    aes_size = 32
    aes_plain = image[:aes_size]
    padding = 16 - len(aes_plain) % 16
    padded = aes_plain + bytes([padding]) * padding
    encryptor = Cipher(algorithms.AES(aes_key), modes.ECB()).encryptor()
    encrypted = encryptor.update(padded) + encryptor.finalize()
    xor_tail = bytes(value ^ xor_key for value in image[-8:])
    data = (
        b"\x07\x08V2\x08\x07"
        + struct.pack("<II", aes_size, len(xor_tail))
        + b"\x00"
        + encrypted
        + image[aes_size:-8]
        + xor_tail
    )

    content, suffix = decode_wechat_image_dat(
        data,
        image_key=aes_key,
        xor_key=xor_key,
    )

    assert content == image
    assert suffix == ".jpg"
