# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-df424694932b0793aaedff791f54d5ac971c24ed551452ee216f10c505396c8d`
- fixture hash: `sha256-cdd5cd85fb616c8f44b236f115a79978bc2dcad4597a177039207ba517f1bddf`
- score hash: `sha256-aa0126960148316d00676d56449adbbb3890c7779f0f994befceae703233a63f`
- bundle hash: `sha256-347767059473ab004a38b670a838864619dcaed77becccd5febd7022df85b865`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df3745ac4e10090248775f0174e4f7f9517bcadad1b8588a0276c1d2f867a57c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-bd5343edc685aee6e0c728855217f96bbf81c471045fefc2d4750c44902866ab |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b4e4888034980e61f04667411710d30ce6f7a049a4df3cfbaab5258f0bd205d5 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e70fc8458fd9f47fb5b023a98aa7adae46950ae4ef356cde1dda4d5d957a01c9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0e36e22e | sha256-d789a270d5dcc2cdc4383a832f0f43d1d25b539dd6c284a80b6b2a322b7246f2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0e36e22e | sha256-d789a270d5dcc2cdc4383a832f0f43d1d25b539dd6c284a80b6b2a322b7246f2 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-1965d1e3 | sha256-67bb476f68e21779b9ac6eccfa505f0a7ec96d50082252648f76d31b7c8847aa |
