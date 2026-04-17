# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b9882f49bf30cb6d948087b310dd1f1c8c43cb51ebde7842866360d6db046b12`
- fixture hash: `sha256-371ddc3cfed0332b92f92e9c2b214fd34bd05f438837cc6562acfdd4c1e2c749`
- score hash: `sha256-6b546240b3d50eb3f37f112c079ec4cd00e68e954334cf1272c945940b0846d2`
- bundle hash: `sha256-1534f4368021cd545a088663b53fa48e2b6e68ba81426042d41d81421fd1f199`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d3958fc805f3776c38e6e687c85563bf09e68cc8dca03392a973d72cef995c7a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-64317ee0a975de97d2bf6038f74ff4daa2b7d2dc8f10aac03318d2485264ba6f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-38f7926e82458872199f1f93f6f418a3092727a8f35f9f6707f12f25a3486681 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-440183b3b4623e499f03d77343b177bbddaafb59c96fab4633c1873cb3614d23 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-531e2678 | sha256-4d74a9ad885bb125354c73bb82965f88c8faa57537dea7a20dca7f8aa87376e2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-531e2678 | sha256-6e00f7f03d76e49f95ec87b2cba82a68728aa4196245a9571eeb7a7b727af4ba |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-934689b1 | sha256-1751ffaebc87d3155e61505fddf3226c4bf16820e451f0e8e177235f915fbc39 |
