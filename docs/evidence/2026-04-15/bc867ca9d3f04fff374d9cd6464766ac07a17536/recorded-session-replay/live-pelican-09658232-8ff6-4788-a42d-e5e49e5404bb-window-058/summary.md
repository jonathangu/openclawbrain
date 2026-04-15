# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70e107aa90463a0c77bc30d344eca5153707641920ed24320747bbd52e05a0e6`
- fixture hash: `sha256-2a5cd5afc4b09fa9beced059043152cd23fab3958640aae8275a1e91138ba120`
- score hash: `sha256-82c2c89db9a300b85ef4314715b79500ee9b32a7e508f484841cb7c462c74727`
- bundle hash: `sha256-42ddb475bbc2219e135a63aeec5e85042835bbc8cc73ca79beefa3ca5e0e624a`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1c146c3106aa3a476acb28a8b075ba9caa0dc741d245d11ea00bbf3c4bbed6c9 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6b13414d985cc3376edffb909603cdfa7aa2c90231e173e77c4a4fb7b3287b46 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8a31df558e84eb37da99da12a398316026ea5d7d739352a0c283926c6fc9e109 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-b3bf1f2082d8e8fd297ffa59c18ab849430cc01c1fdbdf3b11b50151d1ed74be |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ea02d82e | sha256-b893e312cf859f9662fe84b12e9bd8dc18930c4ee2a32f6180e19ccf7f250a11 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ea02d82e | sha256-e9a5383cdabe6829f42d291fd4b03c1a03088947998f3860346d78ffc0c81207 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ea02d82e | sha256-b893e312cf859f9662fe84b12e9bd8dc18930c4ee2a32f6180e19ccf7f250a11 |
