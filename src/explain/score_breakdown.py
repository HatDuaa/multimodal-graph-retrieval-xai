"""Split a fused reranker score into its CLIP and graph parts, then spread the graph part over explanation rows.

z-score fusion (level 1). Level 1 is exact: fuse_tensor computes fused = a * z_clip + (1 - a) * graph.
Level 2 is an approximation: the graph part is a z-score over the top-50 candidates, so it cannot be split exactly
into per-row terms; it is shared out in proportion to each row's raw contribution (channel weight * w * a * sim),
over the channels that are active for the query only.

Raw fusion (fuse_raw, the improvement-phase models): fused = a * CLIP + b * S_obj + c * S_tri, and a channel score is
S = sum_i w_i * sum_j a_ij * sim_ij over the query parts i, so every part is exact (split_raw_score).
"""


def split_final_score(fused, clip_weight, z_clip, graph, contributions, active):
    """Parts and shares of the final score.

    contributions: {channel: [raw contribution per row]}; active: {channel: bool}.
    Shares are part / fused and are None when fused <= 0 (a percentage of a non-positive total is meaningless);
    row parts are None for rows of inactive channels, or when the active raw contributions do not sum above zero.
    """
    clip_part = clip_weight * z_clip
    graph_part = (1 - clip_weight) * graph
    total = sum(value for channel, values in contributions.items() if active[channel] for value in values)

    def share(part):
        return None if part is None or fused <= 0 else part / fused

    row_parts = {channel: [graph_part * value / total if active[channel] and total > 0 else None for value in values]
                 for channel, values in contributions.items()}
    return {"clip_part": clip_part, "graph_part": graph_part,
            "clip_share": share(clip_part), "graph_share": share(graph_part),
            "row_parts": row_parts,
            "row_shares": {channel: [share(part) for part in parts] for channel, parts in row_parts.items()}}


def split_raw_score(fused, clip_weight, clip, terms, fills):
    """Exact parts and shares of fused = a * clip + sum over channels of (channel weight * S) under the raw fusion.

    terms: {channel: [channel weight * w_i * sum_j a_ij * sim_ij per query part]}, which add up to the channel's part.
    fills: {channel: part or None}; when an image has no part a channel can match (S missing), fuse_raw gives it the
    mean S of the candidates that have one, and that filled part (channel weight * mean) stands in for the rows.
    Shares are part / fused, None when fused <= 0, so the parts add up to fused and the shares to 1.
    """
    clip_part = clip_weight * clip
    channel_parts = {channel: (fills.get(channel) if fills.get(channel) is not None else sum(values))
                     for channel, values in terms.items()}
    graph_part = sum(channel_parts.values())

    def share(part):
        return None if part is None or fused <= 0 else part / fused

    return {"clip_part": clip_part, "graph_part": graph_part,
            "clip_share": share(clip_part), "graph_share": share(graph_part),
            "row_parts": {channel: list(values) for channel, values in terms.items()},
            "row_shares": {channel: [share(part) for part in values] for channel, values in terms.items()},
            "fill_parts": dict(fills), "fill_shares": {channel: share(part) for channel, part in fills.items()}}
