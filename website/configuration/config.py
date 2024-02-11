from taipy import Config, Scope
from models.pose import *
climbing_video = Config.configure_data_node(id="video",
                                           storage_type="generic",
                                           scope=Scope.GLOBAL)

pose_data = Config.configure_data_node(id="pose")

predict_pose_task_cfg = Config.configure_task(id="predict_post",
                                              function=predict_pose,
                                              input=climbing_video,
                                              output=pose_data)


scenario_cfg = Config.configure_scenario(id="scenario", task_configs=[predict_pose_task_cfg])