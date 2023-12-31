import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_transformer_decoder(self, center_objects_feature, center_objects_type, obj_feature, obj_mask, obj_pos,
                              map_feature, map_mask, map_pos):
    # intention_query, intention_points = self.get_motion_query(center_objects_type)
    query_content = torch.zeros_like(intention_query)
    # self.forward_ret_dict['intention_points'] = intention_points.permute(1, 0, 2)  # (num_center_objects, num_query, 2)

    num_center_objects = query_content.shape[1]
    num_query = query_content.shape[0]

    center_objects_feature = center_objects_feature[None, :, :].repeat(num_query, 1,
                                                                       1)  # (num_query, num_center_objects, C)

    base_map_idxs = None
    pred_waypoints = intention_points.permute(1, 0, 2)[:, :, None, :]  # (num_center_objects, num_query, 1, 2)
    dynamic_query_center = intention_points

    pred_list = []
    for layer_idx in range(self.num_decoder_layers):
        # query object feature
        obj_query_feature = self.apply_cross_attention(
            kv_feature=obj_feature, kv_mask=obj_mask, kv_pos=obj_pos,
            query_content=query_content, query_embed=intention_query,
            attention_layer=self.obj_decoder_layers[layer_idx],
            dynamic_query_center=dynamic_query_center,
            layer_idx=layer_idx
        )

        # query map feature
        # collected_idxs, base_map_idxs = self.apply_dynamic_map_collection(
        #     map_pos=map_pos, map_mask=map_mask,
        #     pred_waypoints=pred_waypoints,
        #     base_region_offset=self.model_cfg.CENTER_OFFSET_OF_MAP,
        #     num_waypoint_polylines=self.model_cfg.NUM_WAYPOINT_MAP_POLYLINES,
        #     num_base_polylines=self.model_cfg.NUM_BASE_MAP_POLYLINES,
        #     base_map_idxs=base_map_idxs,
        #     num_query=num_query
        # )

        map_query_feature = self.apply_cross_attention(
            kv_feature=map_feature, kv_mask=map_mask, kv_pos=map_pos,
            query_content=query_content, query_embed=intention_query,
            attention_layer=self.map_decoder_layers[layer_idx],
            layer_idx=layer_idx,
            dynamic_query_center=dynamic_query_center,
            use_local_attn=True,
            query_index_pair=collected_idxs,
            query_content_pre_mlp=self.map_query_content_mlps[layer_idx],
            query_embed_pre_mlp=self.map_query_embed_mlps
        )

        query_feature = torch.cat([center_objects_feature, obj_query_feature, map_query_feature], dim=-1)
        query_content = self.query_feature_fusion_layers[layer_idx](
            query_feature.flatten(start_dim=0, end_dim=1)
        ).view(num_query, num_center_objects, -1)

        # motion prediction
        query_content_t = query_content.permute(1, 0, 2).contiguous().view(num_center_objects * num_query, -1)
        pred_scores = self.motion_cls_heads[layer_idx](query_content_t).view(num_center_objects, num_query)
        if self.motion_vel_heads is not None:
            pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query,
                                                                                self.num_future_frames, 5)
            pred_vel = self.motion_vel_heads[layer_idx](query_content_t).view(num_center_objects, num_query,
                                                                              self.num_future_frames, 2)
            pred_trajs = torch.cat((pred_trajs, pred_vel), dim=-1)
        else:
            pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query,
                                                                                self.num_future_frames, 7)

        pred_list.append([pred_scores, pred_trajs])

        # update
        pred_waypoints = pred_trajs[:, :, :, 0:2]
        dynamic_query_center = pred_trajs[:, :, -1, 0:2].contiguous().permute(1, 0,
                                                                              2)  # (num_query, num_center_objects, 2)

    if self.use_place_holder:
        raise NotImplementedError

    assert len(pred_list) == self.num_decoder_layers
    return pred_list