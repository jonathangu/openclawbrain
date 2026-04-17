# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ed2f9f31e28bc4c542ba13fc0a4ccba3e6b6e5db3982235d09f16d62242d7c5e`
- fixture hash: `sha256-c571aef0c0ac7b60f97a81ecefc88f95d1024f6a761836a503482febdda1b1eb`
- score hash: `sha256-cffc5bfe3f43eeed1196492f8d06c22655e07b7766c3db5eff6edec6c4f8f21b`
- bundle hash: `sha256-6bec60098e8c3fe932e83c80ce5a370abec6ed94c48eeeaaffe7ead1d37346ce`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4a371a11e3f0400310e154f8ea3c13a532ee5c397c446eff3697fe01cbdc026c |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-de4c4b5daaf6500b2b18ad63095d551320591000cdb5a1f345a19443d8db69dd |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-e91392e782473d8516d480ebc5e6c19e3f9dad58acccece70ddcde8cf601fec1 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-7219a0c8c533855b8c985d51eede0d1706927a2e490ebbde0c03f6c2684411a4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-2b01ebdc | sha256-6cd065965925ef897e8f1bbe0005fd55e4ba421aa151d400e0f3ff50663496eb |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-2b01ebdc | sha256-380928664f4590908ce088b3da576e68008e7d2c652297cf2a29faa7bf4765b9 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-946b549f | sha256-33051faaa821ab3f34353649c9391e5b1b2e8dabc5ebcadfcb1759edfc1a62a8 |
