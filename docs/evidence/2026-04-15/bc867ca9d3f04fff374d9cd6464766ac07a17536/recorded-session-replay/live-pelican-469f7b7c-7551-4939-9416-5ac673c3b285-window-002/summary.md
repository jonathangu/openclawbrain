# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-53e7f7c2a908bfa01e8a36f987e9389c06b6f1c4270256cec14da19431b1dd8e`
- fixture hash: `sha256-4dd26bce21297c56105a43961b6bacbe27d7812f2b72d27dc4b8b7698e0474b9`
- score hash: `sha256-0274da3a2a08be48b0e4278fe80bf5564356fe102966e2c76d3cbddf9a6e160d`
- bundle hash: `sha256-e2b64d51714df313e3ca650f5085d8fb64b5fc1e42de2b1ecfc591248dc3b26e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aeccda1f8aefa0b00a23d8464e4e2bbf0fb55e8c49bf77bf016cce252f0ffad2 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3e6f0eec3f00d46abc045957e6a956b567132aefc27555e5a7baaead2f4cd40d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f978fba3081c16edbf7996cb05eeb3698c9c7b5abd27930a5829054cc8dcbce4 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d7fba1f9137c4065fe0c57de79a22b4e87decf3932a08c4c5dd8869b34628704 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3dfcbd4e | sha256-2bdb1190ddf62f00b485cab6da55609d785adb288c8d33b0a5dedbbd06df4812 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3dfcbd4e | sha256-f6aadf927781ae4e7885c4351a5d2dab88648c9d6275a00ed66aff5d2161a0d9 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3dfcbd4e | sha256-2bdb1190ddf62f00b485cab6da55609d785adb288c8d33b0a5dedbbd06df4812 |
