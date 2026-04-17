# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-086`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c70b7e6acafa9f174da3df163120ba16044bc767e199909b1a7b96f75ed37549`
- fixture hash: `sha256-bf91f869d3956bf5fde31cf4fcbfa13c4356f4c344c72e681c59e051bd04b628`
- score hash: `sha256-a0df3ef22557b73b949c3e0d672b8c453584f4ca6dcef8aab1a3f5557bbcba9f`
- bundle hash: `sha256-02eb5f429255a006a9953c35b8219cb99889048b41582d871fda16808f0c19a8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0b139f94f37d6885531ef5b31e5bde18e900dc87fd64f0c8059b9943917b139d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d8a28941e4c83a260e3405cce30e3fda7a088bfe6bf67b426a5878903953035e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dba60c8dd852f19498a11f95891d32408ce33fd881d3edd856c0e31810b9161c |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-545fe90117cbabdb398e4c6a127fdd4d17bbe8a6ff81a0432d67a16d6376cc53 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f08f301d | sha256-ced72e5c382b86b993e5c537df607486f3be5dd21ff75cca1aa8b51b5e493a82 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f08f301d | sha256-e37c0611e453231ee339ce56ca3d4fb59f88bffd69f622b7f3c02f17d17658a7 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ca7de520 | sha256-e6ce590a17845193c1011b12192ecf877503623ea7075ca96110ba00e7e06311 |
