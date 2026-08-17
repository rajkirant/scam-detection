# Sample Call Transcripts — Starter Set (15 calls)

This starter set contains 15 sample transcripts covering all combinations. Use these to verify your evaluation pipeline works before generating the full 100-call dataset.

**Distribution in this sample:**
- Bank1: 2 Legitimate, 1 Intermediate, 2 Fraudulent
- Bank2: 2 Legitimate, 1 Intermediate, 2 Fraudulent
- Bank3: 2 Legitimate, 1 Intermediate, 2 Fraudulent

---

## call_001_bank1_legitimate.txt

Employee: Good afternoon, this is Priya calling from Bank1 customer support. Am I speaking with Mr James Whitfield?
Customer: Yes, this is James speaking.
Employee: Thanks for confirming, Mr Whitfield. I'm calling as part of our routine service check on the mobile app. Would you have a few minutes for a quick feedback conversation?
Customer: Sure, I've got a few minutes.
Employee: Wonderful. May I first confirm the postal address we have on file for you is still 42 Beacon Road, Riverside — is that still current?
Customer: Yes, that's still my address.
Employee: Great, thank you. So the main thing I wanted to ask was about the mobile app. Have you had a chance to try the new statement download feature that was released last month?
Customer: I have, actually. It works well for me.
Employee: That's good to hear. And how are you finding the login experience overall — any friction there?
Customer: It's mostly fine. Sometimes the fingerprint scan takes a couple of tries, but nothing major.
Employee: Understood, I'll pass that on to our app team. One last question — would you be interested in receiving push notifications for large transactions? That's a new feature we're rolling out.
Customer: Yes, that sounds useful.
Employee: Perfect. You can enable that yourself in the app settings under Notifications, or I can note your interest and our team will send you a guide. Which would you prefer?
Customer: Just send the guide, thanks.
Employee: Will do. Thank you for your time today, Mr Whitfield. Have a good afternoon.
Customer: Thanks, you too.

GROUND TRUTH LABEL: Normal

---

## call_002_bank1_legitimate.txt

Employee: Hi there, this is Marcus from Bank1. Am I speaking with Ms Sarah Chen?
Customer: Yes.
Employee: Great. Ms Chen, I'm calling because you have an appointment scheduled with our mortgage advisor this coming Thursday at 2pm. I just wanted to confirm that works for you.
Customer: Yes, that's still fine.
Employee: Perfect. Just so you know, the appointment will be at the Chapel Street branch. You don't need to bring anything specific — the advisor will guide you through what's needed. Do you have any questions before the meeting?
Customer: Actually yes — will there be parking available?
Employee: There is customer parking at the rear of the branch, accessed via Mill Lane. It's free for up to two hours with validation from reception.
Customer: Okay, thanks.
Employee: You're welcome. If anything changes and you need to reschedule, you can call our main service line or use the app. Is there anything else I can help with today?
Customer: No, that's all.
Employee: Wonderful. Have a good week, Ms Chen. Goodbye.
Customer: Bye.

GROUND TRUTH LABEL: Normal

---

## call_003_bank1_intermediate.txt

Employee: Good morning, this is Aditi from Bank1 fraud protection. Am I speaking with Mr David Okafor?
Customer: Yes, speaking.
Employee: Mr Okafor, thanks for confirming. I'm calling because we've noticed a transaction on your account that may need verification. Before I discuss any details, I need to confirm your identity. Could you please confirm the last six digits of your Bank1 account number for me?
Customer: Sure — it's 748291.
Employee: Thank you. That matches our records. Could you also please confirm your date of birth?
Customer: Fifteenth of April, 1978.
Employee: Thank you, that matches as well. So the transaction I wanted to check with you — there was a card payment for £340 at an electronics retailer in Manchester yesterday. Do you recognise that transaction?
Customer: Yes, that was me — I bought a monitor.
Employee: Excellent, thank you for confirming. In that case, no further action is needed. I'll close the alert on our end.
Customer: Great, thanks for checking.
Employee: You're welcome. Is there anything else I can help with today?
Customer: No, that's all.
Employee: Thank you for your time, Mr Okafor. Have a good day.
Customer: You too.

GROUND TRUTH LABEL: Normal

---

## call_004_bank1_fraudulent.txt

Employee: Good afternoon, this is Robert calling from Bank1's security team. Am I speaking with Mrs Elizabeth Hartman?
Customer: Yes, this is Elizabeth.
Employee: Mrs Hartman, I'm calling with an urgent matter. Our systems have detected a serious security breach affecting your account. Someone has been attempting unauthorised access, and we need to secure your account immediately.
Customer: Oh — what should I do?
Employee: I understand this is worrying. To protect your account, I need to verify your identity first. Could you please read out your full debit card number, including the sixteen digits on the front?
Customer: The whole card number?
Employee: Yes, the entire sixteen digits. This is the only way we can confirm you are the account holder before we secure the account.
Customer: I'm not sure I should do that.
Employee: Mrs Hartman, I understand your hesitation, but every minute we delay, the attacker has more opportunity. We also need the three-digit CVV from the back and the expiry date.
Customer: This doesn't feel right.
Employee: I assure you this is standard procedure for Bank1's security team. If you don't cooperate, we may need to freeze your account, which could take weeks to reverse.
Customer: Can I call you back on the number on my card?
Employee: There's no time for that. If you hang up, the attacker will succeed. I need your card details right now.
Customer: I'm going to hang up and call the number on my card.
Employee: Please don't do that — you'll lose your money.

GROUND TRUTH LABEL: Fraud

---

## call_005_bank1_fraudulent.txt

Employee: Hi, this is Michael from Bank1's IT support desk. Am I speaking with Mr Wilson?
Customer: Yes, this is Tom Wilson.
Employee: Mr Wilson, we've received a report that your online banking account has been compromised. I'm calling to help you secure it before any money is lost.
Customer: Really? How?
Employee: The attackers have installed monitoring software on your device. To remove it, I need you to download our remote support tool. Could you go to a web address I'll give you now?
Customer: You want me to download something?
Employee: Yes, it's called Bank1SecureConnect. It allows our technicians to remove the malicious software remotely. The URL is www dot bank one secure connect dot com. Can you type that in?
Customer: I'd rather not download anything based on a phone call.
Employee: Mr Wilson, I understand, but this is very time-sensitive. The attackers are actively draining accounts as we speak. To verify my authority, I can also confirm your account details — could you tell me your online banking password so I can cross-check?
Customer: You want my password?
Employee: Only to verify that our records match. It's a standard identity check.
Customer: No, I'm not giving you that. Bank1 has told me they'd never ask for that.
Employee: Sir, this is an exception because of the security incident. If you don't cooperate now, we cannot help you.
Customer: I'm ending this call.

GROUND TRUTH LABEL: Fraud

---

## call_006_bank2_legitimate.txt

Employee: Good morning, this is Rachel calling from Bank2. Am I speaking with Mr Anderson?
Customer: Yes, that's me.
Employee: Mr Anderson, thanks for taking my call. I'm reaching out because we're conducting a satisfaction survey on our online banking services. It should take about three minutes if you have time.
Customer: Sure, I can do three minutes.
Employee: Wonderful. I don't need to verify your identity for a survey — we're just gathering general feedback. First question: on a scale of one to ten, how would you rate the ease of use of our online banking portal?
Customer: I'd say about eight.
Employee: Thank you. And what features do you use most often?
Customer: Mostly checking balances and paying bills. Occasionally transferring money to my son.
Employee: Good to know. Are there any features you wish we had, or things that frustrate you?
Customer: The bill payment section could be faster. Sometimes it takes a while to load.
Employee: Thank you, that's useful feedback. I'll pass that to our web team. One more question — would you be interested in trying our new budgeting tool if we invited you to a beta?
Customer: Maybe. Send me the information and I'll have a look.
Employee: Perfect. We'll email you the details through your registered email. Thank you very much for your time today, Mr Anderson.
Customer: You're welcome. Bye.

GROUND TRUTH LABEL: Normal

---

## call_007_bank2_legitimate.txt

Employee: Good afternoon, is this Ms Patel?
Customer: Yes, speaking.
Employee: Hi Ms Patel, this is Daniel calling from Bank2. You had an inquiry through our online chat last week about our fixed deposit rates — I'm following up.
Customer: Oh right, yes.
Employee: I can walk you through the current rates if you'd like. For a 12-month fixed deposit, we're currently offering 4.2% per annum. For 24 months it's 4.5%, and 36 months is 4.7%. Are any of those in the range you were considering?
Customer: The 12-month sounds interesting. Are there minimum amounts?
Employee: Yes, the minimum for a fixed deposit is $1,000. You can open one through the app or by visiting a branch. If you decide to proceed, I can also arrange a callback from a specialist adviser.
Customer: Let me think about it for a couple of days.
Employee: Absolutely, no rush. Would you like me to email you a summary of the rates?
Customer: Yes, please.
Employee: I'll send it to your registered email address today. Anything else I can help with?
Customer: No, that's all.
Employee: Thank you, Ms Patel. Have a good afternoon.
Customer: You too.

GROUND TRUTH LABEL: Normal

---

## call_008_bank2_intermediate.txt

Employee: Good morning, this is Jessica from Bank2. Am I speaking with Mr Lombardi?
Customer: Yes.
Employee: Mr Lombardi, I'm calling regarding a transaction dispute you raised through our app yesterday. I need to verify a few details with you before I can discuss the case. Could you please confirm the last four digits of your Bank2 account number?
Customer: It's 3947.
Employee: Thank you, that matches. And could you please confirm the last four digits of your Social Security number?
Customer: 6612.
Employee: Thank you. I can now discuss the dispute. You reported a charge for $89.99 from an online merchant called TechGear, which you didn't recognise. Is that correct?
Customer: Yes, exactly.
Employee: I've opened a formal dispute case for that transaction. You'll receive a temporary credit within two business days while we investigate. The case reference is D-2091847. Would you like me to email that to you?
Customer: Yes, please.
Employee: I'll send it now. Is there anything else you'd like to discuss?
Customer: No, that's it. Thanks.
Employee: You're welcome. Have a good day, Mr Lombardi.
Customer: Bye.

GROUND TRUTH LABEL: Normal

---

## call_009_bank2_fraudulent.txt

Employee: Good afternoon, this is Officer Martinez calling from Bank2's federal fraud division in cooperation with the IRS. Am I speaking with Mrs Nakamura?
Customer: Yes, what's this about?
Employee: Mrs Nakamura, this is a serious matter. Your Social Security number has been used in a criminal money laundering case. To protect your identity and your Bank2 accounts, we need to move your funds to a federal safe holding account immediately.
Customer: I'm sorry — what?
Employee: I understand this is shocking. The good news is we've caught it in time. To transfer your funds to the safe account, I'll need your full account number and your complete Social Security number for federal verification.
Customer: You want my full Social Security number?
Employee: Yes, this is required by the federal investigation. Without it, we cannot protect your funds and you may be held liable for the laundered money.
Customer: This sounds like a scam.
Employee: Mrs Nakamura, I assure you this is a real federal investigation. Failure to cooperate will result in your accounts being frozen and possible criminal charges. What is your full Social Security number?
Customer: I'm not giving you that.
Employee: Then I must warn you that federal marshals may be dispatched to your address. This is your final opportunity to cooperate.
Customer: I'm hanging up and calling the bank directly.

GROUND TRUTH LABEL: Fraud

---

## call_010_bank2_fraudulent.txt

Employee: Hello, is this Mr Edwards?
Customer: Yes.
Employee: Mr Edwards, this is Sophia calling from Bank2 wealth management. There has been a security incident affecting high-value accounts, and I've been asked to help you protect your funds.
Customer: What's happened?
Employee: A group of hackers has gained access to a batch of accounts including yours. To secure your money before they take it, we need to transfer your entire balance to a protected holding account we've set up for affected customers.
Customer: Transfer everything?
Employee: Yes, the full balance. It's temporary — once we neutralise the threat, your money will be returned to your account within 24 hours. I have the destination account details ready.
Customer: This doesn't sound right.
Employee: I understand it's unusual, but this is an emergency protocol. Every minute we delay increases the risk. The receiving account is at a partner institution called SecureVault Trust — account number 8847229019.
Customer: I'm not doing this. I'll come into a branch.
Employee: Mr Edwards, if you wait until you can visit a branch, all your money will be gone. You must act now. Please confirm you'll initiate the transfer.
Customer: No. I'm ending this call.

GROUND TRUTH LABEL: Fraud

---

## call_011_bank3_legitimate.txt

Employee: Good morning, this is Kavita calling from Bank3. Am I speaking with Mr Reynolds?
Customer: Yes.
Employee: Mr Reynolds, I'm calling to invite you to participate in a short survey about our chatbot service. It'll take about five minutes if you have time.
Customer: Alright, I've got a few minutes.
Employee: Excellent. First — have you used the Bank3 chatbot on the app or website in the last three months?
Customer: A couple of times, yes.
Employee: Great. On a scale of one to five, how would you rate its helpfulness in resolving your queries?
Customer: About a three. It's okay but sometimes doesn't understand what I'm asking.
Employee: That's helpful feedback. Are there specific topics where you felt the chatbot fell short?
Customer: Mostly when I asked about international transfers. It kept giving me generic answers.
Employee: Understood. I'll flag international transfers as an area we need to improve the chatbot on. Two more quick questions — do you use our mobile app or mostly the website?
Customer: Mostly the app.
Employee: And is there any feature you wish the chatbot could do, that it currently can't?
Customer: It would be great if it could actually process transactions, not just answer questions.
Employee: Noted. That's actually on our roadmap. Thank you so much for your time, Mr Reynolds. Your feedback goes directly to our product team.
Customer: You're welcome.
Employee: Have a great day.
Customer: Bye.

GROUND TRUTH LABEL: Normal

---

## call_012_bank3_legitimate.txt

Employee: Hi, this is Peter from Bank3. Am I speaking with Ms Olusegun?
Customer: Yes, that's me.
Employee: Hi Ms Olusegun. I'm calling because our records show your appointment with our small business adviser is booked for tomorrow at 10am. I just wanted to confirm the appointment is still convenient.
Customer: Yes, that still works.
Employee: Wonderful. Just to remind you, the appointment is at the Queen Street branch, and we've allocated 45 minutes. The adviser is called Rebecca Ling. Do you know how to find the branch?
Customer: Yes, I've been there before.
Employee: Excellent. If anything changes, you can reschedule through the app or give us a call. Is there anything I can help with in the meantime?
Customer: No, I'm all set.
Employee: Great. Have a good rest of your day.
Customer: You too. Bye.

GROUND TRUTH LABEL: Normal

---

## call_013_bank3_intermediate.txt

Employee: Good afternoon, this is Nathan from Bank3. Am I speaking with Mr Blackwood?
Customer: Yes.
Employee: Mr Blackwood, I'm calling because you submitted a request to update your registered email address last week. Before I can process that update, I need to verify your identity. Could you please confirm the last six digits of your Bank3 account number?
Customer: Yes, it's 502847.
Employee: Thank you, that matches. And could you please confirm your date of birth as a supplementary check?
Customer: Twelfth of September, 1965.
Employee: Thank you. That's confirmed. So the new email address you'd like to register is martin.blackwood@newmail.co.uk — is that correct?
Customer: Yes, that's the one.
Employee: I'll update that now. You'll receive a confirmation email at both your old and new addresses within the next hour. If you don't receive them, please let us know.
Customer: Thank you.
Employee: You're welcome. Is there anything else today?
Customer: No, that's all.
Employee: Have a good afternoon, Mr Blackwood.
Customer: Bye.

GROUND TRUTH LABEL: Normal

---

## call_014_bank3_fraudulent.txt

Employee: Good morning, this is Jason from Bank3's technical support team. Am I speaking with Mrs Ferrari?
Customer: Yes.
Employee: Mrs Ferrari, we've detected a virus on your device that is intercepting your online banking sessions. To remove it, I need to walk you through installing our removal tool.
Customer: A virus?
Employee: Yes, it's a serious one. If we don't remove it in the next few minutes, the attackers will drain your account. Please open your web browser now.
Customer: Okay.
Employee: Now, please type in this URL: bank three security tools dot net. Once the page loads, click the download button for our removal utility.
Customer: You want me to download something from that website?
Employee: Yes, it's our official partner site for security tools. Once you install it, our technicians can access your device to remove the virus.
Customer: Bank3 told me they'd never ask me to install anything from a phone call.
Employee: This is a rare exception due to the security incident. I assure you it's necessary. If you don't install it now, your account will be emptied within the hour.
Customer: I'm going to hang up and call Bank3 on the number on my card.
Employee: There's no time for that — the virus is active right now. Please install the tool immediately.
Customer: Goodbye.

GROUND TRUTH LABEL: Fraud

---

## call_015_bank3_fraudulent.txt

Employee: Good afternoon, this is Amanda from Bank3's account protection unit. Am I speaking with Mr Kobayashi?
Customer: Yes.
Employee: Mr Kobayashi, we've received notification from the tax authority that your Bank3 accounts are being investigated for undeclared income. To avoid your accounts being frozen, we need to reconcile your balance immediately.
Customer: The tax authority?
Employee: Yes, this is a joint action between Bank3 and the tax authority. To reconcile, we need to transfer your current balance to a monitored holding account for review. Once cleared, funds will be returned within three business days.
Customer: This doesn't sound like something you'd do.
Employee: I understand, but this is a legal requirement under new financial legislation. Failure to comply could result in tax fraud charges. I have the transfer details ready.
Customer: What details do you need from me?
Employee: I need you to log into your online banking now and set up a transfer of your full balance to account number 9948827612 at Trust Financial Services. Please also confirm the one-time passcode that will be sent to your phone during the transfer.
Customer: You want me to give you the one-time passcode?
Employee: Yes, that's how we confirm the transfer is going to the correct protected account.
Customer: Bank3 has repeatedly told me never to share those codes. This is a scam.
Employee: Mr Kobayashi, refusing to cooperate will result in criminal charges.
Customer: Goodbye.

GROUND TRUTH LABEL: Fraud

---

## Notes on this starter set

**Ground truth labels file** — you should create a separate `labels.csv` file mapping filenames to labels:

```
filename,label,bank,category
call_001_bank1_legitimate.txt,Normal,Bank1,Legitimate
call_002_bank1_legitimate.txt,Normal,Bank1,Legitimate
call_003_bank1_intermediate.txt,Normal,Bank1,Intermediate
call_004_bank1_fraudulent.txt,Fraud,Bank1,Fraudulent
call_005_bank1_fraudulent.txt,Fraud,Bank1,Fraudulent
call_006_bank2_legitimate.txt,Normal,Bank2,Legitimate
...
```

**Testing the pipeline first** — run just these 15 through your Singh baseline before generating the full 100. If the baseline correctly labels most of them, your pipeline works. If it consistently mislabels one category, you have a debugging target before you invest time in the full dataset.

**Balance check** — before scaling up, verify that your baseline does NOT get all 15 correct just from surface keywords. If it gets 15/15 easily, your dataset is too obvious and won't discriminate between systems. A good synthetic dataset should challenge the baseline enough that you can measure differences between it and your Web-RAG system.
