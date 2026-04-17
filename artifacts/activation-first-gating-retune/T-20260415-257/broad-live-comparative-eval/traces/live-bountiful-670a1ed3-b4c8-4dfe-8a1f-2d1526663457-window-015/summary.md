# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-06fea57a3516f2a337e636c80dde5aa0f7b5c4e7b115ef7c15ef4879727a06c9`
- fixture hash: `sha256-0bd1e90ada8a113768901038367ef3359fd513f44e7b3d01e72effd5c2301b57`
- score hash: `sha256-6c01870ae34ac1d502f935ad454c8a8b99aa7aca096ceedaaa69bc4065112069`
- bundle hash: `sha256-b7716b447c6d260e4a57726c88677c49decb26a5c176283a2dd1c1856dfd9014`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5c5ee087496e1c83dd50cdb77e530bafbdd0a3348e86d19deb3da1e266821f9a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bddc09cb7718b8ff11e0e08cfdba7709a86fb1b14228c2bbb6046be9bc95992a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6bfc1e8ef20d9a366a4ce89aaab08ee80d218c029a7119a44ebc3673c2eaf964 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-1427e3678092bba2390579848996c2e597944e9c5a920166028faebe457602b4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9bf4ce1a | sha256-0876587e3c88ae0bcc9ed11af54dab0d5b22f1ca2a6e7eaeb6a0975ff7f85661 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9bf4ce1a | sha256-3adc8bfb46c8f2e94fcd3eb95c69f4763d4c2d8dea3973c0e61a259e0b27dafd |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-345ac823 | sha256-e1648d48cc68069201b7710512cedc662d75880b42e93a56cff0be4f4e4b53ee |
