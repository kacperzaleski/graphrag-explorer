# Judge calibration sheet

Label each answer as correct / partial / incorrect against the reference, WITHOUT looking at the judge. Then write one JSON line per answer to `eval/manual_labels.jsonl`, e.g.

`{"qid": "q01", "config": "graph", "label": "correct"}`

## q03 (vector, multi_hop)
**Q:** Which major Austrian bank founded in 1855 acquired a majority stake in Oberbank in 1933?

**Reference:** Creditanstalt

**Answer:** The major Austrian bank founded in 1855 that acquired a majority stake in Oberbank in 1933 is the Creditanstalt (CA) [creditanstalt#000, oberbank#001].

## q08 (graph, multi_hop)
**Q:** Which bank merged on October 1, 2005, with the postal savings bank that was owned by the Austrian Post Office?

**Reference:** BAWAG

**Answer:** The bank that merged with the Österreichische Postsparkasse (the postal savings bank owned by the Austrian Post Office) on October 1, 2005, was BAWAG (Bank für Arbeit und Wirtschaft).

## q14 (vector, lookup)
**Q:** Who was the first President of BayernLB when it was founded in 1972?

**Reference:** Karl Theodor Jacob

**Answer:** The first President of BayernLB when it was founded in 1972 was Karl Theodor Jacob.

## q22 (graph, lookup)
**Q:** What role did Dan Quayle take on when he joined Cerberus in 1999?

**Reference:** Chairman of the company's Global Investments Division

**Answer:** When Dan Quayle joined Cerberus in 1999, he took on the role of chairman of the company's Global Investments Division.

## q27 (vector, corpus_wide)
**Q:** Who have been the owners of BAWAG over time?

**Reference:** Socialist trade unions and the consumer cooperatives' procurement company (founding), later ÖGB and Konsum; BayernLB (46% from 1995); ÖGB as sole owner (2004); Cerberus (90% from 2006) with Österreichische Post, Generali and Wüstenrot; today GoldenTree (21.8%) and T. Rowe Price (5.6%) as largest shareholders of the listed BAWAG Group.

**Answer:** BAWAG's ownership has changed several times throughout its history. Initially, it was held by socialist trade unions and the Großeinkaufsgesellschaft für österreichische Consumvereine (40% each) [bawag#001]. Later, the Austrian Trade Union Federation (ÖGB) held 70% and the Konsum retail cooperative chain held 30% [bawag#001]. After Konsum went bankrupt in 1995, Bayerische Landesbank acquired shares, increasing its stake to 46% [bawag#002]. In 2004, the ÖGB bought the 46% stake from Bayerische Landesbank, becoming the sole owner [bawag#003]. In 2006, Cerberus acquired 90% of BAWAG [bawag#005]. Other shareholders at that time included Österreichische Post (5%), Generali (2-3%), Wüstenrot-Gruppe (1%), and Hannes Androsch [bawag#005]. As of 2022, the US hedge fund GoldenTree was the main shareholder, though it reduced its stake by more than half [bawag#005]. As of 2017, other major shareholders included T. Rowe Price (5.6%) [bawag#000].

## q28 (graph, corpus_wide)
**Q:** Which authorities supervise Austrian banks?

**Reference:** The European Central Bank via European Banking Supervision (directly for significant institutions), the Financial Market Authority (FMA) and the Oesterreichische Nationalbank.

**Answer:** Austrian banks are supervised by the Financial Market Authority (FMA) and the European Central Bank (ECB). The FMA is Austria's integrated financial supervisory authority, responsible for the supervision of credit institutions in a complementary role with the ECB Banking Supervision [financial-market-authority-austria#000]. Additionally, the Oesterreichische Nationalbank (OeNB) participates in European banking supervision as a member of the Supervisory Board of the European Central Bank, alongside the FMA [austrian-national-bank#000].

## q10 (vector, multi_hop)
**Q:** Which Munich-based bank bought the shares of the shareholder Konsum after it went bankrupt in 1995?

**Reference:** Bayerische Landesbank

**Answer:** The Bayerische Landesbank, which is a bank based in Munich, bought the shares of the shareholder Konsum after it went bankrupt in 1995.

## q01 (graph, multi_hop)
**Q:** Which integrated financial supervisory authority participates in European banking supervision as a member of the Supervisory Board of the European Central Bank alongside the Oesterreichische Nationalbank?

**Reference:** Financial Market Authority

**Answer:** The Financial Market Authority (Austria) (FMA) is the integrated financial supervisory authority that participates in European banking supervision as a member of the Supervisory Board of the European Central Bank, alongside the Oesterreichische Nationalbank [austrian-national-bank#000, financial-market-authority-austria#000].

## q23 (vector, lookup)
**Q:** Which American company did BAWAG buy the piano manufacturer Bösendorfer from in 2001?

**Reference:** Kimball International

**Answer:** In 2001, BAWAG bought the piano manufacturer Bösendorfer from the American company Kimball International.

## q17 (graph, lookup)
**Q:** Which Austrian ministry holds the right to consent to individual regulations of the Financial Market Authority (FMA)?

**Reference:** Ministry of Finance

**Answer:** The Ministry of Finance (Austria) holds the right to consent to individual regulations of the Financial Market Authority (FMA).
