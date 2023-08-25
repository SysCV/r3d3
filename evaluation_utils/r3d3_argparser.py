import argparse

from r3d3.process import GraphType
from r3d3.covis_graph import CorrelationMode
from r3d3.frame_buffer import CompletionMode
from r3d3.startup_filter import ImageMode


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='Path to dataset and completion network configuration .yaml file')
    parser.add_argument("--n_workers", type=int, default=1, help='# dataloader workers')
    parser.add_argument("--training_data_path", default=None, help='Save results as completion training samples')
    parser.add_argument("--prediction_data_path", default=None, help='Save results in datafolder')

    parser.add_argument("--r3d3_weights", default="/media/datadrive/droid.pth")
    parser.add_argument("--r3d3_image_size", type=int, nargs="+", default=[384, 640], help='Image dimensions')
    parser.add_argument("--r3d3_image_mode", choices=list(ImageMode), default=ImageMode.BGR,
                        help='If images are proc. as RGB or BGR')
    parser.add_argument("--r3d3_scale", type=float, default=15.0, help="R3D3 works at metric-scale / scale")
    parser.add_argument("--r3d3_completion_mode", choices=list(CompletionMode), default=CompletionMode.ALL,
                        help='Frames which should be completed')
    parser.add_argument("--r3d3_buffer_size", type=int, default=-1, help='Buffer size - Default: Use optm. window')
    parser.add_argument("--r3d3_n_warmup", type=int, default=3, help='# warmup timesteps')
    parser.add_argument("--r3d3_filter_thresh", type=float, default=1.75)
    parser.add_argument("--r3d3_depth_init", type=float, default=15.0)
    parser.add_argument("--r3d3_frame_thresh", type=float, default=2.25)
    parser.add_argument("--r3d3_init_motion_only", action='store_true')
    parser.add_argument("--r3d3_iters_init", type=int, default=8, help='# initialization GRU+DBA iterations')
    parser.add_argument("--r3d3_iters1", type=int, default=4, help='# update GRU+DBA iterations')
    parser.add_argument("--r3d3_iters2", type=int, default=2, help='# update GRU+DBA iterations if prev. frame not rm.')
    parser.add_argument("--r3d3_optm_window", type=int, default=1)
    parser.add_argument("--r3d3_disable_comp_inter_flow", action="store_true", help='Diable inter-cam rot.-flow comp.')
    parser.add_argument("--r3d3_corr_impl", type=CorrelationMode, choices=list(CorrelationMode),
                        default=CorrelationMode.VOLUME, help='Correlation volume implementation')
    parser.add_argument("--r3d3_n_edges_max", type=int, default=-1, help='Max. number of graph-edges. -1: no limit')
    parser.add_argument("--r3d3_graph_type", type=GraphType, choices=list(GraphType), default=GraphType.STATIC,
                        help='Which graph implementation to use')
    parser.add_argument("--r3d3_proximity_thresh", type=float, default=16.0,
                        help='Prox. thresh. for Droid-SLAM graph constr.')
    parser.add_argument("--r3d3_nms", type=int, default=1, help='Non-max-suppr. for Droid-SLAM graph constr.')
    parser.add_argument("--r3d3_max_age", type=int, default=25, help='Max edge age for Droid-SLAM graph constr.')
    parser.add_argument("--r3d3_ref_window", type=int, default=5)
    parser.add_argument("--r3d3_dt_intra", type=int, default=3, help='Temp. edge time window for graph constr.')
    parser.add_argument("--r3d3_dt_inter", type=int, default=2, help='Spat. temp. edge time window for graph constr.')
    parser.add_argument("--r3d3_r_intra", type=int, default=2, help='Max. radius for temporal graph constr.')
    parser.add_argument("--r3d3_r_inter", type=int, default=2, help='Radius for spat-temporal graph constr.')

    return parser.parse_args()
