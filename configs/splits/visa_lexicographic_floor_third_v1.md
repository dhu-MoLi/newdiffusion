# VisA defect-type split `visa_lexicographic_floor_third_v1`

Fair-replay protocol for official-500/1000. Do not use VisA `meta.json` or `split_csv/`.

## Rules

- Categories with `image_anno.csv`, lexicographic. Extra-A pilot = first 4: candle, capsules, cashew, chewinggum.
- Skip `label==normal`, comma-multi-label, empty mask.
- Extra-A types: single-label, `n>=6`, lexicographic first 4.
- Few-shot = lexicographic first `floor(n/3)` of that type. Held-out = the rest.
- `test/good` = lexicographic last `max(1, floor(N/3))` normals; remainder `train/good`.
- DualAnoDiff eval puts **all** selected-type images into `test/<slug>/`, including few-shot.
- Slug: spaces → `_`. Informal collapsed arm `defect` is a different split; see JSON.

## Extra-A official types (use these to match official-500/1000)

| category | type | slug | n | few | hold | few-shot IDs |
| --- | --- | --- | ---: | ---: | ---: | --- |
| candle | chunk of wax missing | `chunk_of_wax_missing` | 12 | 4 | 8 | `000 001 002 003` |
| candle | damaged corner of packaging | `damaged_corner_of_packaging` | 18 | 6 | 12 | `010 011 012 013 014 016` |
| candle | different colour spot | `different_colour_spot` | 15 | 5 | 10 | `040 041 042 043 045` |
| candle | extra wax in candle | `extra_wax_in_candle` | 9 | 3 | 6 | `060 061 062` |
| capsules | bubble | `bubble` | 30 | 10 | 20 | `000 001 002 003 004 005 006 007 008 009` |
| cashew | burnt | `burnt` | 10 | 3 | 7 | `001 003 004` |
| cashew | corner or edge breakage | `corner_or_edge_breakage` | 14 | 4 | 10 | `012 013 014 015` |
| cashew | different colour spot | `different_colour_spot` | 9 | 3 | 6 | `025 026 027` |
| cashew | middle breakage | `middle_breakage` | 6 | 2 | 4 | `032 033` |
| chewinggum | chunk of gum missing | `chunk_of_gum_missing` | 16 | 5 | 11 | `016 017 020 023 024` |
| chewinggum | corner missing | `corner_missing` | 13 | 4 | 9 | `005 025 026 027` |
| chewinggum | scratches | `scratches` | 9 | 3 | 6 | `066 067 068` |
| chewinggum | similar colour spot | `similar_colour_spot` | 19 | 6 | 13 | `075 076 077 078 079 082` |

## All VisA categories (same rules; extra types kept in the copied tree)

Non-pilot `selected` means the same `n>=6` / first-4 cap, not that official-500 used them.

| category | extra-A pilot | type | slug | type-cap | n | few | hold |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| candle | yes | chunk of wax missing | `chunk_of_wax_missing` | selected | 12 | 4 | 8 |
| candle | yes | damaged corner of packaging | `damaged_corner_of_packaging` | selected | 18 | 6 | 12 |
| candle | yes | different colour spot | `different_colour_spot` | selected | 15 | 5 | 10 |
| candle | yes | extra wax in candle | `extra_wax_in_candle` | selected | 9 | 3 | 6 |
| candle | yes | foreign particals on candle | `foreign_particals_on_candle` | eligible | 12 | 4 | 8 |
| candle | yes | wax melded out of the candle | `wax_melded_out_of_the_candle` | eligible | 12 | 4 | 8 |
| candle | yes | weird candle wick | `weird_candle_wick` | eligible | 9 | 3 | 6 |
| capsules | yes | bubble | `bubble` | selected | 30 | 10 | 20 |
| cashew | yes | burnt | `burnt` | selected | 10 | 3 | 7 |
| cashew | yes | corner or edge breakage | `corner_or_edge_breakage` | selected | 14 | 4 | 10 |
| cashew | yes | different colour spot | `different_colour_spot` | selected | 9 | 3 | 6 |
| cashew | yes | middle breakage | `middle_breakage` | selected | 6 | 2 | 4 |
| cashew | yes | same colour spot | `same_colour_spot` | eligible | 10 | 3 | 7 |
| cashew | yes | small holes | `small_holes` | eligible | 20 | 6 | 14 |
| cashew | yes | small scratches | `small_scratches` | eligible | 15 | 5 | 10 |
| cashew | yes | stuck together | `stuck_together` | eligible | 6 | 2 | 4 |
| chewinggum | yes | chunk of gum missing | `chunk_of_gum_missing` | selected | 16 | 5 | 11 |
| chewinggum | yes | corner missing | `corner_missing` | selected | 13 | 4 | 9 |
| chewinggum | yes | scratches | `scratches` | selected | 9 | 3 | 6 |
| chewinggum | yes | similar colour spot | `similar_colour_spot` | selected | 19 | 6 | 13 |
| fryum | no | burnt | `burnt` | selected | 9 | 3 | 6 |
| fryum | no | corner or edge breakage | `corner_or_edge_breakage` | selected | 8 | 2 | 6 |
| fryum | no | different colour spot | `different_colour_spot` | selected | 14 | 4 | 10 |
| fryum | no | fryum stuck together | `fryum_stuck_together` | selected | 20 | 6 | 14 |
| fryum | no | middle breakage | `middle_breakage` | eligible | 18 | 6 | 12 |
| fryum | no | similar colour spot | `similar_colour_spot` | eligible | 12 | 4 | 8 |
| fryum | no | small scratches | `small_scratches` | eligible | 6 | 2 | 4 |
| macaroni1 | no | chip around edge and corner | `chip_around_edge_and_corner` | selected | 17 | 5 | 12 |
| macaroni1 | no | different colour spot | `different_colour_spot` | selected | 15 | 5 | 10 |
| macaroni1 | no | similar colour spot | `similar_colour_spot` | selected | 21 | 7 | 14 |
| macaroni1 | no | small cracks | `small_cracks` | selected | 11 | 3 | 8 |
| macaroni1 | no | small scratches | `small_scratches` | eligible | 16 | 5 | 11 |
| macaroni2 | no | breakage down the middle | `breakage_down_the_middle` | n<6 | 1 | 0 | 1 |
| macaroni2 | no | color spot similar to the object | `color_spot_similar_to_the_object` | selected | 23 | 7 | 16 |
| macaroni2 | no | different color spot | `different_color_spot` | selected | 12 | 4 | 8 |
| macaroni2 | no | small chip around edge | `small_chip_around_edge` | selected | 19 | 6 | 13 |
| macaroni2 | no | small cracks | `small_cracks` | selected | 7 | 2 | 5 |
| macaroni2 | no | small scratches | `small_scratches` | eligible | 16 | 5 | 11 |
| pcb1 | no | bent | `bent` | selected | 14 | 4 | 10 |
| pcb1 | no | melt | `melt` | selected | 46 | 15 | 31 |
| pcb1 | no | missing | `missing` | selected | 17 | 5 | 12 |
| pcb1 | no | scratch | `scratch` | selected | 14 | 4 | 10 |
| pcb2 | no | bent | `bent` | selected | 12 | 4 | 8 |
| pcb2 | no | melt | `melt` | selected | 48 | 16 | 32 |
| pcb2 | no | missing | `missing` | selected | 17 | 5 | 12 |
| pcb2 | no | scratch | `scratch` | selected | 14 | 4 | 10 |
| pcb3 | no | bent | `bent` | selected | 19 | 6 | 13 |
| pcb3 | no | melt | `melt` | selected | 38 | 12 | 26 |
| pcb3 | no | missing | `missing` | selected | 16 | 5 | 11 |
| pcb3 | no | scratch | `scratch` | selected | 20 | 6 | 14 |
| pcb4 | no | burnt | `burnt` | n<6 | 2 | 0 | 2 |
| pcb4 | no | damage | `damage` | selected | 7 | 2 | 5 |
| pcb4 | no | extra | `extra` | selected | 12 | 4 | 8 |
| pcb4 | no | missing | `missing` | selected | 19 | 6 | 13 |
| pcb4 | no | scratch | `scratch` | n<6 | 2 | 0 | 2 |
| pcb4 | no | wrong place | `wrong_place` | n<6 | 2 | 0 | 2 |
| pipe_fryum | no | burnt | `burnt` | selected | 15 | 5 | 10 |
| pipe_fryum | no | corner and edge breakage | `corner_and_edge_breakage` | selected | 15 | 5 | 10 |
| pipe_fryum | no | different colour spot | `different_colour_spot` | selected | 10 | 3 | 7 |
| pipe_fryum | no | similar colour spot | `similar_colour_spot` | selected | 20 | 6 | 14 |
| pipe_fryum | no | small scratches | `small_scratches` | eligible | 18 | 6 | 12 |
| pipe_fryum | no | stuck together | `stuck_together` | eligible | 10 | 3 | 7 |

Full image lists, skipped multi-label rows, and collapsed-arm IDs are in the JSON.
