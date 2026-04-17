# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6cd785628fb3c34642dd7b4a701799a6e96acb06e347a7bf1d01cd4950a8de4c`
- fixture hash: `sha256-8169ace4aebbc5a4a546b5c0d2bdc7c5a395f1f1630a066be79c7f63594673d2`
- score hash: `sha256-161340ab3237b9aad489d47a27ed22e1362f4d55724f98c8dc494526b6aefc88`
- bundle hash: `sha256-2a5aac9d217ed520acc808b83d3e7c4d6c31e90d19a442a6fa9c611a589727d1`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf6fbe7613b07ee3e659c5ab0ce2fe9e83640dc0dbe17b255f0c268784354a36 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-307341f69f102b49f6646ebeea0891301c7b7e47bbe52b83934e1e55bee9bbb8 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-75fd1b35f3c6bea53284ead73117979e901a4fbcc3241bcc58500b7b63eea456 |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-e2047b313ac6e4f49ebd8f3bd9d3f72d4a3fb9ec57e8438b21c1e0cdcd1fd634 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-2caebda2 | sha256-ec5838d599db5e2150b98a917c8f808957056d7ee207400e070f01fa5df65dc6 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-2caebda2 | sha256-2c4b1899277a03d111a8d50cf425c76fd66fcc086d77807dc85b261233de986c |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-e336d769 | sha256-98289fc9e6331bb8c62a124928f58447c26d7f2349f5f1964779b3ffadf15f5f |
