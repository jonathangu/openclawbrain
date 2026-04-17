# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-077`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7fb0ffa9f4e8d70c4fc8ddccf35cb362423daf7116804c6734e40b9d0f4296bf`
- fixture hash: `sha256-810d932d8ac4f8eed98f074f82298ad7f5b0354d5fdf19533c533df6c21240d2`
- score hash: `sha256-4e69f77fa550f34006367d5e29ea1311979803a62c4fc58191cbf489cdbc2f86`
- bundle hash: `sha256-773c407d66a6445fcdb75cb63ff8f3221dea0869e16018ba5af3ef73c581ea7d`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fe2cee15d245f859cb5315bfc802316abb7874a3bff97839e84f3440b5d4a896 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b86665edd82089dc89193587598a52949e55c92d9bf3d2a36501c8d29a76d937 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e99a06e4b6682a2d2440ef99e2f9e9bc418fc91862f1b141c0d72b12e81b2200 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-434954911c08179772f504bb6a9e30bfb663f240b8ae2a0a259e8dc1be919115 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-515b0d21 | sha256-acb5153235c5b3a24d682875ce4dc521eead0fca68fa2eaf899361be63a65613 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-515b0d21 | sha256-f88aec376d116e4cb9f1ca3f93746b55541b08dc48a4e33d6ae9cb0d3377427c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-864b166e | sha256-1c3dc5449a10384a82206d4abc0ceb6b285ce02ca856af84e0d0625289f42761 |
