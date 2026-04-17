# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-149`
- winner mode: `graph_prior_only`
- trace hash: `sha256-299806353ab465c5dc0556cb46d4c0ddab82caef7c74016e1f229b80f14988f5`
- fixture hash: `sha256-6a01dd18700c95a1ef47fa69bd96f40af05494c628d07e9816bb8fa24129ae15`
- score hash: `sha256-a1109417b8b703d9e5451b22d92c18e9a833b1b78afeb69a3908dbbeb2bdd44b`
- bundle hash: `sha256-2e3391c47c2d05d6f1cd358b2fe5b9d04aff0ed8f76c5c1c67a64ad79dc130c3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fd2b5dc7e86f33e7f35222bc8995c5714891af6e48c0b188589cfd85f30ab7cd |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a2513fef3940f746fe42eeaae6ef572eaac4169c7218b2e6a6eed99fe589de1d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-007896ba8572d59c6bd53ce1cea4a258e979a7d9aa83a70542bb9fe9352eeca1 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-714372bf3daef5e312b6ce31a14a14e8cbf47a8dda9b1553d370a6a5633d5535 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4f131bcb | sha256-a95c664a8be85805e19094795f34662d8c590f39dd48ddba98cead0c8430fa25 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4f131bcb | sha256-00205ab6059756e8da36e6ce10197634bb68466a7c927b2ac376a47bdaa418c3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f4f01b38 | sha256-c2117de028d6de45747406597bfa5672bd1ee029e479fd580ca6f3667c529b07 |
