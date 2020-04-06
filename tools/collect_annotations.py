import pathlib
import os
import xml.etree.ElementTree as ET

data_dir = "./data/raw"

class AnnotationInfo(object):
    def __init__(self, image_path, img_filename, annotations):
        self.image_path = image_path
        self.image_filename = img_filename
        self.annotations = annotations

    def __repr__(self):
        return "{} {}".format(self.image_path, str(self.annotations))


def parse_xml(xml_file_path, images_dir):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    # print(root)

    filename = root.find("filename").text
    full_image_path = os.path.join(images_dir, filename)
    
    # print(root.filename)

    bboxes = []
    for object_node in root.findall("object"):
        bndbox = object_node.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        xmax = int(bndbox.find("xmax").text)
        ymin = int(bndbox.find("ymin").text)
        ymax = int(bndbox.find("ymax").text)
        # print(xmin, xmax, ymin, ymax)
        bboxes.append((xmin, ymin, xmax, ymax))
    return AnnotationInfo(full_image_path, filename, bboxes)

def get_annotations_from_dir(dir_path : str, images_dir):
    path = pathlib.Path(dir_path)

    annotations = []
    for annotation in path.glob("*.xml"):
        annotation_info = parse_xml(annotation, images_dir)
        annotations.append(annotation_info)
    return annotations

def get_annotations(base_dir):
    path = pathlib.Path(data_dir)    
    
    all_annotations = []
    for annotations_dir in path.glob("*_ann"):
        images_dir = str(annotations_dir).replace("_ann", "")
        annotations = get_annotations_from_dir(annotations_dir, images_dir=images_dir)
        all_annotations.extend(annotations)
    return all_annotations

def main():
    # xml_path = "data/raw/Ci_ychn7ga0_ann/vlcsnap-2020-03-20-12h53m27s066.xml"
    # filename, bboxes = parse_xml(xml_path)
    # print (filename, bboxes)
    # return 
    all_annotations = get_annotations(data_dir)
    return
    # for file in os.listdir(data_dir):


if __name__ == "__main__":
    main()