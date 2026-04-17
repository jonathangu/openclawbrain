# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-056`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3b194659082568a82c511a7152ca31a2b1b95a8940775e0d8501ad2641699262`
- fixture hash: `sha256-818172b532c3157150cdaf4f843fa921402c9f435a9b49f1a0bba05b616c0656`
- score hash: `sha256-03f0683cdfd12418f593701866d7e0959a9956c96dcaad60c6cc6fa73dcf46e5`
- bundle hash: `sha256-0d062afa3a6be2a095dd3573e18a9ccee47f9627d09e6c069921bbca69469ce1`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3c006aa8496cda3a74dc0aceaf43d36eb374dd8330caeef238bfc730df80da87 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6e4cd21fa07cbc1bb0cbadaf8668b6bde76eacf75e69469cd3329593f58bb398 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-06a4967f9db7a55741baaba9f8e36d00daefdb8ecb28c115da89fd272730c99e |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-bc0743aa476491c2c11de09ba665496f1bfc83bf9a5a9ed48e6324af6a31948c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b5106df8 | sha256-8167842979cf3e163cfd4c9a8bafc4a5a45453fd4202d7b12923d2ba25cb85f8 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b5106df8 | sha256-8fe4300e159dd86cbc167fbb31043e7e997898c22730b4bb753849de07cca874 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-0040cc81 | sha256-882016c815a5e9b452e4abd6b3a7d8ca4ea7aa20226c81a8356b581b50cef193 |
