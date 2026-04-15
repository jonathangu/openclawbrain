# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dd29f792fbfbc606fc0ae81485babcea7a498bf8f85b66de2333918434925117`
- fixture hash: `sha256-a84a33537b8d24e458443c5c6b1cbd9d02b490a8b56c8f49f8509184e51ddc87`
- score hash: `sha256-95448f4c011169da66e5f0534ec63e8ca7211ed4711bec5364431277163165e4`
- bundle hash: `sha256-d952e0b2c2b8e91d1176a3407c7295ff627f7c1729040e35873ad5bdb8a6d899`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d6181c2140140f5786a710721c2d0cc92976577da480a328a542e8b790bc4990 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3287da01e64c2936d427606e774d4cf267b3d685824b58852bae660e41f7ea62 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2dac43dede79cc07c5ab3612ac42ba47a5dd34ae24f411ade6ade70cd78d0ae |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d5b27ed7c101a8ca0100cb0b4c8cbada7f423c02ce4ef4ba20414c98cd1340fb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6d43c066 | sha256-2264883b88069f70549bf325a334abc873f48e924bcf5441bd367910f3b6a887 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6d43c066 | sha256-26f1f546ad32d18348894ca0f466594808ff76d63a46e08cfd6bcc08677a81c3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-6d43c066 | sha256-2264883b88069f70549bf325a334abc873f48e924bcf5441bd367910f3b6a887 |
