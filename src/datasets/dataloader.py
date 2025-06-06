class FederatedDataloader:
    """Pass in a client id and return a dataloader for that client
    """

    def __init__(self, data_dir, client_list, split, batch_size, max_num_elements_per_client):
        pass

    def get_client_dataloader(self, client_id, idx):
        raise NotImplementedError

    def __len__(self):
        """Return number of clients."""
        raise NotImplementedError

    def dataset_name(self):
        raise NotImplementedError

    def get_loss_and_metrics_fn(self):
        # loss_fn: return a torch scalar (autodiff enabled)
        # metrics_fn: return an OrderedDict with keys 'loss', 'accuracy', etc.
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError


class ClientDataloader:
    """Dataloader for a client
    """

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError