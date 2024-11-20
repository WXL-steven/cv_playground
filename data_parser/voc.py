from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Tuple, List
import xml

from torch.utils.data import Dataset

from data_parser.utils import FileCache


@dataclass(slots=True)
class VOC2012BBoxSample:
    image_path: Optional[Path] = field(default=None)
    image_valid: Optional[bool] = field(default=False)
    image_height: Optional[int] = field(default=0)
    image_width: Optional[int] = field(default=0)

    annotation_path: Optional[Path] = field(default=None)
    num_objects: Optional[int] = field(default=0)


class VOC2012BBoxDataset(Dataset):
    DEFAULT_IMAGES_DIR = "JPEGImages"
    DEFAULT_ANNOTATIONS_DIR = "Annotations"

    ACCEPTED_IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")
    ACCEPTED_ANNOTATION_EXTENSIONS = (".xml",)

    @staticmethod
    def _check_and_get_paths(
            split_file_path: Union[str, Path],
            root_dir: Optional[Union[str, Path]] = None,
            images_dir: Optional[Union[str, Path]] = None,
            annotations_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[Path, Path, Path]:
        # 验证参数组合的合法性
        if root_dir is None and (images_dir is None or annotations_dir is None):
            raise ValueError(
                "Either root_dir or both images_dir and annotations_dir must be specified"
            )

        # 转换所有路径为Path对象
        split_file_path = Path(split_file_path)
        if root_dir is not None:
            root_dir = Path(root_dir)

        # 合成路径
        if images_dir is None:
            images_dir = root_dir / VOC2012BBoxDataset.DEFAULT_IMAGES_DIR
        else:
            images_dir = Path(images_dir)

        if annotations_dir is None:
            annotations_dir = root_dir / VOC2012BBoxDataset.DEFAULT_ANNOTATIONS_DIR
        else:
            annotations_dir = Path(annotations_dir)

        # 存在性检查
        if not split_file_path.is_file():
            raise FileNotFoundError(f"Task file not found: {split_file_path}")

        if not images_dir.is_dir():
            raise FileNotFoundError(f"Image folder not found: {images_dir}")

        if not annotations_dir.is_dir():
            raise FileNotFoundError(f"Annotation folder not found: {annotations_dir}")

        return split_file_path, images_dir, annotations_dir

    @staticmethod
    def _load_split_file(split_file_path: Path) -> List[str]:
        sample_names = []
        with split_file_path.open() as f:
            for line in f:
                if line is None:
                    continue
                line = line.strip()
                if line == '':
                    continue
                sample_names.append(line)
        return sample_names

    def _scan_images(self, images_dir: Path, sample_names: List[str]) -> List[VOC2012BBoxSample]:
        samples = []
        # 生成缓存
        images_dir_cache = FileCache(directory=images_dir, recursive=False, max_cache=20000)
        if len(images_dir_cache) == 0:
            RuntimeError("No files found in images directory")

        for sample_name in sample_names:
            matched_files: tuple[Path, ...] = images_dir_cache.find_matches(filename=sample_name)
            valid_files = []
            for file_path in matched_files:
                # 检查扩展名
                ext = file_path.suffix.lower()[1:]  # 去掉点号
                if ext not in self.ACCEPTED_IMAGE_EXTENSIONS:
                    continue

                # 检查文件大小
                if file_path.stat().st_size <= 10:
                    continue

                valid_files.append((
                    file_path,
                    len(file_path.name),  # 文件名长度
                    self.ACCEPTED_IMAGE_EXTENSIONS.index(ext)  # 扩展名优先级
                ))

            if not valid_files:
                continue

            # 按文件名长度和扩展名优先级排序
            valid_files.sort(key=lambda x: (x[1], x[2]))
            samples.append(VOC2012BBoxSample(image_path=valid_files[0][0]))

        return samples

    def __init__(
            self,
            split_file_path: Union[str, Path],
            root_dir: Optional[Union[str, Path]] = None,
            images_dir: Optional[Union[str, Path]] = None,
            annotations_dir: Optional[Union[str, Path]] = None
    ) -> None:
        # 实例特性类型声明:
        self.split_file_path: Path
        self.images_dir: Path
        self.annotations_dir: Path
        self.sample_names: List[str]

        # 超类实例化
        super().__init__()

        # 处理路径
        self.split_file_path, self.images_dir, self.annotations_dir = self._check_and_get_paths(
            split_file_path, root_dir, images_dir, annotations_dir
        )

        # 将实例名加载到内存
        self.sample_names = self._load_split_file(self.split_file_path)

        # TODO: 考虑将此设置为手动调用可选功能
        # 扫描全部实例的图像以确保其存在并生成统计
        self.samples = self._scan_images(self.images_dir, self.sample_names)
