from config import cfg

path = cfg.paths["imagenetloc"]
# read the cdv file

annotation_dict = {}
with open(f"{path}/LOC_val_solution.csv") as f:
    lines = f.readlines()
    for line in lines[1:]:
        img, annotation_line = line.split(",")
        # annotation is a line like this :
        # class_name1 xmax xmin ymax ymin class_name2 xmax xmin ymax ymin ...
        # parse the annotation
        annotations = annotation_line[:-2].split(" ")
        annotations = [annotations[i:i+5] for i in range(0, len(annotations), 5)]
        # compare the class name of the annotation of the same line
        
        annotation_classes = [annotation[0] for annotation in annotations]
        # if all the name are the same
        if len(set(annotation_classes)) == 1:
            annotation_dict[img] = annotation_classes[0]
        else:
            print(f"Image {img} has more than 1 class")
            print(annotation_classes)
            break
