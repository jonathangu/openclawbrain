# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3ae37b035b14582a9db80eca92cf9ea284e1a083da7e40c15c27766593c501ab`
- fixture hash: `sha256-c1a05a74d2fece7febaade02118c0528463204c7c70c4fc0e050990958f60a91`
- score hash: `sha256-7848b40b4258bea7eafbdf0c9967ebf6cdd9c5a8d3d896c9535a758b01e7da32`
- bundle hash: `sha256-d77667c64f40d28097eaeca24fc874b7f6ecf7afbd15e83d7431745ac7b5375b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5bfccd81fb07da2148e74d03332b298ae8343e32f9c89c9de3c815764af2fb42 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9de3c9449b65c9d9ff905c87cc968e3ad0404926cde4b0e1f3a6a5ecb055db58 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a95a5b89479c39b716e574e903dd06f469bb61134a484cc30a857da4379086a5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-59cabb232586187a09c7b19caf886b5d6d6f1aa37c73ddb51b89f338726be23f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-54b6150b | sha256-98f86593a0407c1732c32ef44262488bbc39edeefcdfc3d978be9a9f450bb7a6 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-54b6150b | sha256-fc569bd1084eae3396414419ce5e01af28661cb010b042685deb0175dcebf39d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0d1144f8 | sha256-e3c98595535c51ba9ef5be7fa2ae9d8750d30d1b875a240bd67bad03e1cbf05d |
