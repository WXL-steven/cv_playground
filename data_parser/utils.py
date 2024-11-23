from pathlib import Path
from typing import Union, Tuple
from collections import defaultdict
import os


class FileCache:
    def __init__(
            self,
            directory: Union[str, Path],
            max_cache: int = 200000
    ) -> None:
        """
        初始化文件缓存系统

        Args:
            directory: 要搜索的目录
            max_cache: 最大缓存条目数，小于1表示不限制
        """
        self.directory = Path(directory)
        self.max_cache = max_cache
        self.cache = defaultdict(tuple)
        self._build_cache()

    @staticmethod
    def _get_key(filename: str) -> str:
        """
        从文件名获取缓存键
        将文件名按点分割取第一部分，去除首尾空格并转为小写
        """
        return filename.split('.')[0].strip().lower()

    def _build_cache(self) -> None:
        """建立文件索引缓存"""
        files = []
        counter = 0

        # 使用os.scandir()只扫描一层目录
        with os.scandir(self.directory) as entries:
            for entry in entries:
                if entry.is_dir():  # 只处理文件,忽略文件夹
                    continue

                files.append(Path(entry.path))

                # 检查是否超过最大缓存限制
                counter += 1
                if 0 < self.max_cache <= counter:
                    raise ValueError(
                        f"Directory contains more than {self.max_cache} files. "
                        "Please increase max_cache limit or narrow your search scope."
                    )

        # 建立缓存
        temp_cache = defaultdict(list)
        for file_path in files:
            key = self._get_key(file_path.name)
            temp_cache[key].append(file_path)

        # 将列表转换为元组以节省内存并防止修改
        self.cache = {k: tuple(v) for k, v in temp_cache.items()}

    def find_matches(self, filename: str) -> Tuple[Path, ...]:
        """
        查找匹配的文件

        Args:
            filename: 要查找的文件名（不含扩展名）

        Returns:
            包含所有匹配文件路径的元组
        """
        key = self._get_key(filename)
        return self.cache.get(key, tuple())

    def __len__(self) -> int:
        """返回缓存的文件总数"""
        return sum(len(matches) for matches in self.cache.values())

    def __str__(self) -> str:
        """返回缓存状态的字符串表示"""
        return (f"FileCache(directory='{self.directory}', "
                f"cached_files={len(self)})")


def _test():
    import time
    t0 = time.time()
    cache = FileCache(r'../datasets/VOC2012/JPEGImages')
    print(f"Cache created in {(time.time() - t0) * 1000:.2f} ms")
    t0 = time.time()
    matches = cache.find_matches('2007_000783')
    print(f"Find matches in {(time.time() - t0) * 1000:.2f} ms")
    print(f"Cache length: {len(cache)}")
    print(f"Cache: {matches}")


if __name__ == '__main__':
    _test()
