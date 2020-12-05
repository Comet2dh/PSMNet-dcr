from models.PSM_init import PSM_init
from models.PSM_dcr import PSM_dcr
from models.loss import model_loss

__models__ = {
    "PSM_init": PSM_init,
    "PSM_dcr": PSM_dcr
}
