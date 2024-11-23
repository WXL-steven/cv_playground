import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Tuple, List, Callable, Set

from torch.utils.data import Dataset
import cv2
from lxml import etree

from data_parser.utils import FileCache


@dataclass(slots=True)
class VOC2012BBoxSample:
    sample_name: str = field()

    image_path: Optional[Path] = field(default=None)
    image_available: Optional[bool] = field(default=None)
    image_height: int = field(default=-1)
    image_width: int = field(default=-1)

    annotation_path: Optional[Path] = field(default=None)
    annotation_available: Optional[bool] = field(default=None)
    num_objects: int = field(default=-1)
    num_broken: int = field(default=-1)


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
    def _load_split_file(split_file_path: Path) -> List[VOC2012BBoxSample]:
        samples = []
        with split_file_path.open() as f:
            for line in f:
                if line is None:
                    continue
                line = line.strip()
                if line == '':
                    continue
                samples.append(
                    VOC2012BBoxSample(sample_name=line)
                )
        return samples

    @staticmethod
    def _parse_boxes(element: etree.Element) -> Tuple[str, int, int, int, int]:
        cls_node = element.xpath('./name')[0].text
        xmin_node = element.xpath('./bndbox/xmin')[0].text
        ymin_node = element.xpath('./bndbox/ymin')[0].text
        xmax_node = element.xpath('./bndbox/xmax')[0].text
        ymax_node = element.xpath('./bndbox/ymax')[0].text

        return cls_node, int(xmin_node), int(ymin_node), int(xmax_node), int(ymax_node)

    def scan_images(self, progress_callback: Optional[Callable[[], None]] = None) -> None:
        # 生成缓存
        images_dir_cache = FileCache(directory=self.images_dir, max_cache=20000)

        if len(images_dir_cache) == 0:
            raise RuntimeError("No files found in images directory")

        for sample_instance in self.samples:
            sample_name = sample_instance.sample_name
            matched_files: Tuple[Path, ...] = images_dir_cache.find_matches(filename=sample_name)
            file_path: Optional[Path] = None
            for matched_path in matched_files:
                file_name = matched_path.stem
                file_extension = matched_path.suffix.lower()
                if file_extension in VOC2012BBoxDataset.ACCEPTED_IMAGE_EXTENSIONS and file_name == sample_name:
                    file_path = matched_path
                    break

            if file_path is None:
                sample_instance.image_available = False
                continue

            sample_instance.image_path = file_path

            img_array = cv2.imread(file_path.as_posix())
            if img_array is None:
                sample_instance.image_available = False
                continue

            sample_instance.image_height = img_array.shape[0]
            sample_instance.image_width = img_array.shape[1]
            sample_instance.image_available = True

            if progress_callback is not None:
                progress_callback()

        return

    def scan_annotations(self, progress_callback: Optional[Callable[[], None]] = None) -> None:
        annotations_dir_cache = FileCache(directory=self.annotations_dir, max_cache=20000)

        if len(annotations_dir_cache) == 0:
            raise RuntimeError("No files found in annotations directory")

        for sample_instance in self.samples:
            if not sample_instance.image_available:
                continue

            sample_name = sample_instance.sample_name
            matched_files: Tuple[Path, ...] = annotations_dir_cache.find_matches(filename=sample_name)
            file_path: Optional[Path] = None
            for matched_path in matched_files:
                file_name = matched_path.stem
                file_extension = matched_path.suffix.lower()
                if file_extension in VOC2012BBoxDataset.ACCEPTED_ANNOTATION_EXTENSIONS and file_name == sample_name:
                    file_path = matched_path
                    break

            if file_path is None:
                sample_instance.annotation_path = None
                continue

            sample_instance.annotation_path = file_path

            # 解析为ElementTree对象
            tree = etree.parse(file_path.as_posix())

            # 处理图像名
            filename_node = tree.xpath('/annotation/filename')
            if len(filename_node) < 1:
                # 无法读取到图像名
                sample_instance.annotation_available = False
                continue

            filename = filename_node[0].text
            if filename.lower() != sample_instance.image_path.name.lower():
                # 图像名不匹配
                sample_instance.annotation_available = False
                continue

            sample_instance.annotation_available = True

            # 解析BBox
            obj_nodes = tree.xpath('/annotation/object')
            sample_instance.num_objects = len(obj_nodes)
            sample_instance.num_broken = 0
            for obj_node in obj_nodes:
                try:
                    cls = self._parse_boxes(obj_node)[0]
                    self.class_names.add(cls)
                except (ValueError, IndexError) as e:
                    print(e)
                    sample_instance.num_objects -= 1
                    sample_instance.num_broken += 1
                    continue

            if progress_callback is not None:
                progress_callback()

        return

    def __init__(
            self,
            split_file_path: Union[str, Path],
            root_dir: Optional[Union[str, Path]] = None,
            images_dir: Optional[Union[str, Path]] = None,
            annotations_dir: Optional[Union[str, Path]] = None,
            auto_scan: bool = True
    ) -> None:
        # 超类实例化
        super().__init__()

        # 实例特性类型声明及初始化:
        self.split_file_path: Path
        self.images_dir: Path
        self.annotations_dir: Path
        self.samples: List[VOC2012BBoxSample]
        self.class_names: Set[str] = set()

        # 处理路径
        self.split_file_path, self.images_dir, self.annotations_dir = self._check_and_get_paths(
            split_file_path, root_dir, images_dir, annotations_dir
        )

        # 将实例名加载到内存
        self.samples = self._load_split_file(self.split_file_path)

        # TODO: 考虑将此设置为手动调用可选功能
        # 扫描全部实例的图像以确保其存在并生成统计
        if auto_scan:
            t0 = time.time()
            self.scan_images()
            t = time.time() - t0
            num_undefined = sum(1 for sample in self.samples if sample.image_path is None)
            num_unavailable = sum(1 for sample in self.samples if not sample.image_available)
            print(f"Scanned {len(self.samples)} samples for images in {t * 1000:.2f} ms, "
                  f"{num_undefined} undefined, "
                  f"{num_unavailable} unavailable, "
                  f"avg. {t * 1000 / len(self.samples):.2f} ms per item, ")

            # 扫描标注
            t0 = time.time()
            self.scan_annotations()
            t = time.time() - t0
            num_undefined = sum(1 for sample in self.samples if sample.annotation_path is None)
            num_unavailable = sum(1 for sample in self.samples if not sample.annotation_available)
            num_objects = sum(sample.num_objects for sample in self.samples)
            num_broken = sum(sample.num_broken for sample in self.samples)
            print(f"Scanned {len(self.samples)} samples for annotations in {t * 1000:.2f} ms, "
                  f"got {num_objects} objects, "
                  f"{num_undefined} undefined, "
                  f"{num_unavailable} unavailable, "
                  f"{num_broken} broken, "
                  f"avg. {t * 1000 / len(self.samples):.2f} ms per item, "
                  f"avg. {num_objects / len(self.samples):.2f} objects per object.")


def _test():
    dataset = VOC2012BBoxDataset(
        split_file_path=r"../datasets/VOC2012/ImageSets/Main/train.txt",
        root_dir=r"../datasets/VOC2012/"
    )


if __name__ == "__main__":
    _test()
