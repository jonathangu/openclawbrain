# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4b9d58238866be4c30cb67001ed41c476bb074abc457ce91f27bdf2a95087dda`
- fixture hash: `sha256-93a191f41c9134f7fb1b39f4120c598d79722f0fdf720a1c60726eeea45f85a7`
- score hash: `sha256-b0cabba356f4094e9041ee2a590b74aa5eb84e0dcf401a11b36f707c47522ab9`
- bundle hash: `sha256-e7f2433379b3a2f152b802fbc581a8b4ba722d65a5a7998d54ecf626cd548736`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-53ac170bbdfe31610a82a7fea6a20f739ad327e9856e23aa713b46f86601ea52 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-9993afe39c8427d5a5239ec2cbfd3db17fe2b1274e9d2b5ca8a6d20088f2b962 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-411a3a4404f6c20e51d5c2070636c22950eaa09a1db3944cc66900844cc04d31 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-d61703619aa238b8ceb00816556b4486d59453313e40a96f5163dcf7bae9c3f7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-cf9305aa | sha256-d6e445f5389c331ee379205440e1cad393d836ac62b63c273333a880399279ed |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-cf9305aa | sha256-3df843192b62d76ce9c8bc930d0766d0bb804af540c171bc3425e69d57f3e37f |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-ae0ad487 | sha256-3c6dd0a34f399de1d5ce523680e99a6a85abe38a8d328ae17c1af82d3d1e23ce |
