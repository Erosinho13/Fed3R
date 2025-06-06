from .mobilenetv2 import mobilenetv2, load_classifier


def get_model_from_args(args, device):
    if args.dataset.lower() == 'gldv2' or args.dataset.lower() == 'inaturalist':
        if args.model_name in ['mobilenetv2']:
            print('Runnign gldv2 with pretrained mobilenetv2')
            model = mobilenetv2(num_classes=2028 if args.dataset.lower() == 'gldv2' else 1203, return_features=False)
            fed3r_cls = getattr(args, 'load_fed3r_classifier', False)
            if fed3r_cls:
                load_classifier(model, 'saved_models/fed3r_classifiers', fed3r_cls)
                print(f"Loaded Fed3R classifier (exp {fed3r_cls})")
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    return model.to(device)
