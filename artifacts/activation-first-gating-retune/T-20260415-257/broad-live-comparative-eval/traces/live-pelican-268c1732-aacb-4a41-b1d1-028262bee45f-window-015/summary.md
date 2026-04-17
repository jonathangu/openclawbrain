# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae32f07fbe5a45648ccbd2d0869190b2cb3596e4fc7c3e1299ef7f3819e0b838`
- fixture hash: `sha256-b830296ad0e542a07399e1e822eb8c0691a725d5f9135851e63c87d0c1b12ee0`
- score hash: `sha256-037831b2c22768e496524551c9faaf8c430a0631b687b4c35c6c180cc1e0491a`
- bundle hash: `sha256-b79819f9bb045bd3e1f84dddfb260764c09cba62f9e87933526cb6203bb102cb`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4f883a174ba2c6d9b8e46baf2069a63ced4f1f39ba1f842535f04648f9481662 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a618c66b0953d763ae48bb626fbcbbc0a0b7f5d458eb894ed771bc75fe95f248 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-05d77fb15ae18c4df6fe929f770419d52990a9b061267ad44d4cdc15cc016134 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d37308cc6a202752a8d4588b9849f901bb4f9e59cd438cf919473909f5e4f51e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a5219a43 | sha256-5449597bd0699180559453afd4fbee66a09d8224e8881fd91f01ea08f3f412a7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a5219a43 | sha256-4be2c9b0d41ac3a9d0218edbdabbc9d687abba525a6f7c2a6ee13b0f37093f4f |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-461710bc | sha256-5344cc6beef54ee7b44c0fe4d03535220e72f1c3680be733c30945fd058edd4e |
