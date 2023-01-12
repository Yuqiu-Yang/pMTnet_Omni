import pytest


from pMTnet_Omni.background_tcr_loaders import background_tcr_dataset_class,\
    background_tcr_dataloader_class, \
    background_species_tcr_dataset_class, \
    background_species_tcr_dataloader_class


@pytest.mark.parametrize("species, chain, load_embedding, load_size, expected", [
    ("human", "alpha", False, 5, [2, 3]),
    ("human", "beta", False, 1, [1, 3]),
    ("mouse", "alpha", True, 5, [2, 35]),
    ("mouse", "beta", True, 1, [1, 35])
])
def test_background_tcr_dataset_class(species, chain, load_embedding, load_size, expected):
    dataset = background_tcr_dataset_class(data_dir="./tests/test_data/background_tcrs/",
                                           species=species,
                                           chain=chain,
                                           load_embedding=load_embedding,
                                           load_size=load_size)
    n_rows = len(dataset)
    n_cols = dataset[[0]].shape[1]
    assert (n_rows == expected[0]) and (n_cols == expected[1])


@pytest.mark.parametrize("batch_size, expected", [
    (3, [3, 3, 3, 3, 3]),
    ([4, 5, 6], [4, 5, 6])
])
def test_background_tcr_dataloader_class(batch_size, expected):
    dataset = background_tcr_dataset_class(data_dir="./tests/test_data/background_tcrs/",
                                           species="human",
                                           chain="alpha",
                                           load_embedding=False,
                                           load_size=2)
    dataloader = background_tcr_dataloader_class(background_tcr_dataset=dataset,
                                                 batch_size=batch_size)
    n_rows = []
    if isinstance(batch_size, int):
        for i in range(5):
            tcrs = next(iter(dataloader))
            n_rows.append(tcrs.shape[0])
        assert n_rows == expected
    else:
        for tcrs in dataloader:
            n_rows.append(tcrs.shape[0])
        assert n_rows == expected

######################################
# Species


@pytest.mark.parametrize("species, load_embedding, load_size, expected", [
    ("human", False, 5, [4, 6]),
    ("human", False, 1, [2, 6]),
    ("mouse", True, 5, [4, 70]),
    ("mouse", True, 1, [2, 70])
])
def test_background_species_tcr_dataset_class(species, load_embedding, load_size, expected):
    dataset = background_species_tcr_dataset_class(data_dir="./tests/test_data/background_tcrs/",
                                                   species=species,
                                                   load_embedding=load_embedding,
                                                   load_size=load_size)
    n_rows = len(dataset)
    n_cols = dataset[([0], [0])].shape[1]
    assert (n_rows == expected[0]) and (n_cols == expected[1])


@pytest.mark.parametrize("batch_size, expected", [
    (3, [3, 3, 3, 3, 3]),
    ([4, 5, 6], [4, 5, 6])
])
def test_background_species_tcr_dataloader_class(batch_size, expected):
    dataset = background_species_tcr_dataset_class(data_dir="./tests/test_data/background_tcrs/",
                                                   species="human",
                                                   load_embedding=False,
                                                   load_size=2)
    dataloader = background_species_tcr_dataloader_class(background_species_tcr_dataset=dataset,
                                                         batch_size=batch_size)
    n_rows = []
    if isinstance(batch_size, int):
        for i in range(5):
            tcrs = next(iter(dataloader))
            n_rows.append(tcrs.shape[0])
        assert n_rows == expected
    else:
        for tcrs in dataloader:
            n_rows.append(tcrs.shape[0])
        assert n_rows == expected
