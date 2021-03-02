from torch.utils.data.dataloader import default_collate


def collate_with_none(batch):
    """
    A Pytorch collate_fn that can handle none elements
    :param batch:
        The input batch
    :return:
        The input batch is Nones filtered out, if all batch data
        is corrupted, returns an empty batch
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return ([], [], [])
    return default_collate(batch)
