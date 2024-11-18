from suprank.losses.arcface_loss import ArcFaceLoss
from suprank.losses.blackbox_ap_loss import BlackBoxAPLoss
from suprank.losses.calibration_loss import CalibrationLoss
from suprank.losses.cluster_loss import ClusterLoss
from suprank.losses.csl_loss import CSLLoss
from suprank.losses.fast_ap_loss import FastAPLoss
from suprank.losses.proxy_nca_pp_loss import ProxyNCAppLoss
from suprank.losses.ranking_loss import SmoothHAPLoss, SupHAPLoss, SmoothAPLoss, SupAPLoss, SupNDCGLoss, SmoothNDCGLoss, SupRecallLoss, SmoothRecallLoss
from suprank.losses.softbin_ap_loss import SoftBinAPLoss
from suprank.losses.triplet_loss import TripletLoss
from suprank.losses.xbm_loss import XBMLoss


__all__ = [
    'ArcFaceLoss',
    'BlackBoxAPLoss',
    'CalibrationLoss',
    'ClusterLoss',
    'CSLLoss',
    'FastAPLoss',
    'ProxyNCAppLoss',
    'SmoothHAPLoss', 'SupHAPLoss', 'SmoothAPLoss', 'SupAPLoss', 'SupNDCGLoss', 'SmoothNDCGLoss', 'SupRecallLoss', 'SmoothRecallLoss',
    'SoftBinAPLoss',
    'TripletLoss',
    'XBMLoss',
]
