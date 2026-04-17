# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-df541968ca52654e5efa48a1a6713bb4511f8366d389ef30e36174b0478a0f72`
- fixture hash: `sha256-10a1d9d424d59bf74d6edef2d25c3d9864b38e04e75b6ff4b28dfed92245cd1e`
- score hash: `sha256-c374c06f007597912f1d3ddb500eb485ffc02bfbd31bdbde702ac26d1513fb9d`
- bundle hash: `sha256-679886b6acff5051893143486c457cc9f3b56a5300eafaf5450ed14777fb4ca9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe456f8f9f99c14a6c26ae3cfe1240fa644752e272eded6c0df3fca37912d301 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b45e26abfa1ffbdf2ed84944a972f3730089a67b142449f52bf83ac7d32c1519 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6af6b16dec7c9bd8247c2baba82e904613ef0343588fac7936196d1bee476e57 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-9f93ce698e758ab835572d1eea9d76da9fa37e6239bcf11f8c9e981f166c90e8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b8a12013 | sha256-7c512f8f851087313b4d03976d37e3f4c8e40dc150ad619b69a21fdd26a1ad41 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b8a12013 | sha256-69172681e56a986c8d84c7e14e6148a43be52013784dcde455026f4956538f82 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-602002cc | sha256-2a477ef3d21e7719efe57844cae8c5bc07c34c78e0d7b2f4e2a6b4b9d44010a4 |
