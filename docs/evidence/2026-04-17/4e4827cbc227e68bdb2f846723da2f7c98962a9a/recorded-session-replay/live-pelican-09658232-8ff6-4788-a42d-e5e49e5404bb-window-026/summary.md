# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-026`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b6b834033c55b0c1fa4b1b699a1d8df0de8cfc6135b0ef6c168d6806431da077`
- fixture hash: `sha256-da3f651615d0891ce7f18953e9f938d1c01935c73ef17cad6fec24cf102a80c5`
- score hash: `sha256-658e641a149cb9b3bf61c11971bbe8e50c55ae90a41f0b640100f0139b572837`
- bundle hash: `sha256-e2c29ba11f3377b1c427c3ade469f49ac4a694307428a26d00c4b401de87abc0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7bb32e7c3cacde4e347cf09cbdabeddff358a8d0a73f0b6ff3c688033dc4ad3f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-94ac9cb95567577865d19a738f2a81cbac76ecef14f5341151884d25b02a0ebf |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a30e895d73b083bc9306b91cef21dc3475a8a15ec8a598d5ce9b2798b8368fdd |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ada467b123ee9b8d47e5c5d9ba40af350389cb52f23e611984258b1932df45b5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-45713494 | sha256-69178a8afb51d92754ac88fef1bb78c568895af91ec9c08939c7f513d545d35d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-45713494 | sha256-69178a8afb51d92754ac88fef1bb78c568895af91ec9c08939c7f513d545d35d |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-2a172b37 | sha256-2c9a3c239a7e6d06fc5baa8a8aed879ea202f0b0f1db7e3951da105363a8b258 |
