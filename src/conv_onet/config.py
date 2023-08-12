from src.conv_onet import models


def get_model(cfg, patch_size=1, conf_feats=7, num_sensors=1):
    """
    Return the network model.

    Args:
        cfg (dict): imported yaml config.

    Returns:
        decoder (nn.module): the network model.
    """

    dim = cfg["data"]["dim"]
    middle_grid_len = cfg["grid_len"]["middle"]
    fine_grid_len = cfg["grid_len"]["fine"]
    color_grid_len = cfg["grid_len"]["color"]
    c_dim = cfg["model"]["c_dim"]  # feature dimensions
    pos_embedding_method = cfg["model"]["pos_embedding_method"]

    total_feats = patch_size**2 * conf_feats
    col_feats = patch_size**2 * 3

    decoder = models.decoder_dict["uncle"](
        dim=dim,
        conf_dim=total_feats,
        col_conf_dim=col_feats,
        c_dim=c_dim,
        middle_grid_len=middle_grid_len,
        fine_grid_len=fine_grid_len,
        color_grid_len=color_grid_len,
        pos_embedding_method=pos_embedding_method,
        num_sensors=num_sensors,
    )

    return decoder
