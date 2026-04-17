# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-7ade65ed-f8fd-4d4d-8c8f-77ff9531b42b-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1cc116d5a5a3e4268eee5081d6d597a83a2afaebb6c2529b01952ad2f45437c1`
- fixture hash: `sha256-ccd8a0f1240cc7f92941ab2c1ede0327e4ed0a420f6a51ec4c81e0437c7d59e2`
- score hash: `sha256-51d02ea67c8a0bac55f9f7ce5122b4eff94552a7d96d3171e3f8ee53e9405602`
- bundle hash: `sha256-0e682ab317ae1d17a8205abebcb8d0c0542546daed0195d5f9aa0ebfe31bbc54`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a97ee439ef356b4483f5735f34054ec24021480ea2dadec6ac22262eafbebd17 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c584aaa5f8ce1df75343169463c7907682b282a584b745503e624ba418340b23 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5ffde908a603cc31b88eced4a2ce1a61c01f92cf1dc8f08ac90b0c6f70d37a0e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-948f7e1efad7a4d63347fdc97f8b5a3c6b37f2f0410fce82982197c89894e140 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1ff681f1 | sha256-689050637dac936cc958e02e1394d6e90d12c08fc7797ec37b524a26443d40b8 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1ff681f1 | sha256-d92fad5578c9bbd44c9b198e85e4e871ade1f294eab7b98cdb4783e6d133543a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-285e7c78 | sha256-765ebace9c50ec90ee0c53aa860719f2a9f920506a11a9d57561eaa1f198c4f3 |
