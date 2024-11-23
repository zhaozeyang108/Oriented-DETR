
from .dota import build as build_dota
def get_coco_api_from_dataset(dataset):
    return dataset.coco
def build_dataset(image_set, args):

    if args.dataset_file == 'dota':
        return build_dota(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
