use super::{Ask, Item};

fn choice(out: &mut Vec<Item>, domain: &str, instructions: &str, options: &[&str], state: String, gold: &str) {
    assert!(options.contains(&gold), "{domain}: {gold} not in {options:?}");
    out.push(Item {
        domain: domain.into(),
        state,
        ask: Ask::Choice {
            instructions: instructions.into(),
            options: options.iter().map(|s| (*s).to_string()).collect(),
            gold: gold.into(),
        },
    });
}

fn noul(out: &mut Vec<Item>, domain: &str, state: String, statement: String, gold: bool) {
    out.push(Item {
        domain: domain.into(),
        state,
        ask: Ask::Noul { statement, gold },
    });
}

fn score(out: &mut Vec<Item>, domain: &str, instructions: &str, levels: &[&str], state: String, gold: &str) {
    assert!(levels.contains(&gold), "{domain}: {gold} not in {levels:?}");
    out.push(Item {
        domain: domain.into(),
        state,
        ask: Ask::Score {
            instructions: instructions.into(),
            levels: levels.iter().map(|s| (*s).to_string()).collect(),
            gold: gold.into(),
        },
    });
}

pub fn build() -> Vec<Item> {
    let mut out = Vec::new();
    intent(&mut out);
    topic(&mut out);
    entailment(&mut out);
    abstain(&mut out);
    routing(&mut out);
    expense(&mut out);
    priority(&mut out);
    sentiment(&mut out);
    news(&mut out);
    document(&mut out);
    industry(&mut out);
    country(&mut out);
    cuisine(&mut out);
    profession(&mut out);
    meeting(&mut out);
    product(&mut out);
    weather(&mut out);
    hr(&mut out);
    compare(&mut out);
    risk(&mut out);
    folder(&mut out);
    let base = out.clone();
    for prefix in ["From the queue: ", "Filed this morning: "] {
        for item in &base {
            if out.len() >= 1000 {
                return out;
            }
            let mut next = item.clone();
            next.state = format!("{prefix}{}", item.state);
            out.push(next);
        }
    }
    assert!(out.len() >= 1000, "only {} items", out.len());
    out.truncate(1000);
    out
}

fn intent(out: &mut Vec<Item>) {
    let q = "What is the sender trying to do?";
    let opts = ["reschedule a meeting", "share a status update", "request a document", "acknowledge a message", "none"];
    let events = ["budget review", "design critique", "staff meeting", "roadmap review", "vendor call", "hiring debrief", "quarterly planning", "security review"];
    let days = ["Friday at 2pm", "Monday morning", "next Wednesday", "Thursday after lunch", "tomorrow at 11"];
    for event in events {
        for day in days {
            choice(out, "intent", q, &opts, format!("Can we move the {event} to {day}? I have a conflict."), "reschedule a meeting");
        }
    }
    let systems = ["staging", "billing", "search", "login", "reports", "mobile"];
    let metrics = ["error rate is flat", "latency is unchanged", "no new alerts fired"];
    for system in systems {
        for metric in metrics {
            choice(out, "intent", q, &opts, format!("The {system} deploy finished. {metric} and the release is live."), "share a status update");
        }
    }
    let docs = ["signed statement of work", "updated price list", "insurance certificate", "board deck", "security questionnaire"];
    let when = ["before the kickoff", "by Friday", "before the client call"];
    for doc in docs {
        for deadline in when {
            choice(out, "intent", q, &opts, format!("Please send me the {doc} {deadline}."), "request a document");
        }
    }
    let thanks = ["Thanks for the intro.", "Got the calendar hold.", "Received the packet.", "Appreciate the summary.", "Thanks, that answers it."];
    let follow = ["I will read the brief this afternoon.", "I will reply after I check with finance.", "I have what I need for now.", "I will look at the numbers tonight."];
    for a in thanks {
        for b in follow {
            choice(out, "intent", q, &opts, format!("{a} {b}"), "acknowledge a message");
        }
    }
}

fn topic(out: &mut Vec<Item>) {
    let q = "Which subject is this paragraph about?";
    let opts = ["finance", "hiring", "facilities", "product support"];
    let firms = ["Northwind", "the retail unit", "the European office", "the services group"];
    let facts = [
        ("revenue was $4.2 million, up 6%", "finance"),
        ("gross margin held at 41%", "finance"),
        ("cash on hand covers nine months of payroll", "finance"),
        ("the audit found no material misstatement", "finance"),
    ];
    for firm in firms {
        for (fact, gold) in facts {
            choice(out, "topic", q, &opts, format!("At {firm}, {fact}."), gold);
        }
    }
    let roles = ["senior accountant", "payroll specialist", "accounts payable clerk", "financial analyst", "controller"];
    let closes = ["the 15th", "Friday", "the end of the month"];
    for role in roles {
        for close in closes {
            choice(out, "topic", q, &opts, format!("We are opening a {role} role. Applications close on {close}."), "hiring");
        }
    }
    let places = ["third floor", "loading dock", "east stairwell", "lobby"];
    let issues = ["HVAC unit failed overnight", "badge reader is offline", "water leak was reported", "elevator is out of service"];
    for place in places {
        for issue in issues {
            choice(out, "topic", q, &opts, format!("The {issue} on the {place}. That area is closed until it is repaired."), "facilities");
        }
    }
    let features = ["reports page", "invoice export", "admin console", "mobile checkout"];
    let symptoms = ["customers cannot export CSV", "the save button returns an error", "search results are empty", "the page will not load"];
    for feature in features {
        for symptom in symptoms {
            choice(out, "topic", q, &opts, format!("On the {feature}, {symptom} after yesterday's release."), "product support");
        }
    }
}

fn entailment(out: &mut Vec<Item>) {
    let dates = ["June 1", "September 15", "January 10", "March 31", "November 1"];
    let notices = ["30 days", "45 days", "60 days"];
    for date in dates {
        for notice in notices {
            let state = format!("The contract renews on {date} unless either party gives {notice}' written notice. No notice has been sent.");
            noul(out, "entailment", state.clone(), format!("The contract is set to renew on {date}."), true);
            noul(out, "entailment", state, "A party has already cancelled the renewal.".into(), false);
        }
    }
    let people = [("Jordan", "designer"), ("Sam", "recruiter"), ("Alex", "writer"), ("Riley", "analyst"), ("Casey", "teacher")];
    for (name, job) in people {
        let state = format!("Only managers in the payroll group can approve overtime. {name} is a {job} and is not in that group.");
        noul(out, "entailment", state.clone(), format!("{name} can approve overtime."), false);
        noul(out, "entailment", state, format!("{name} is not a payroll manager."), true);
    }
    let goods = [("chairs", 80, 100), ("laptops", 12, 20), ("desks", 5, 8), ("monitors", 30, 40)];
    for (item, got, total) in goods {
        let left = total - got;
        let state = format!("The vendor delivered {got} of the {total} {item}. The remaining {left} ship next Tuesday.");
        noul(out, "entailment", state.clone(), format!("Some of the {item} have not arrived yet."), true);
        noul(out, "entailment", state, format!("Every one of the {item} has already arrived."), false);
    }
}

fn abstain(out: &mut Vec<Item>) {
    let payroll = ["run payroll", "correct a tax withholding", "issue a bonus", "none"];
    let unrelated = [
        "Please book a conference room for six people on Monday morning.",
        "The cafeteria serves soup on Wednesdays.",
        "Attached is the agenda for the design critique.",
        "The train to the client site is delayed twenty minutes.",
        "Please water the plants in the reception area.",
        "The all-hands slides are in the shared folder.",
        "Can someone bring name tags to the lobby?",
        "The parking garage is full after 9am.",
        "Lunch is at noon in the courtyard.",
        "The guest wifi password changed this morning.",
    ];
    for text in unrelated {
        choice(out, "abstain", "Which payroll action is requested? Choose none if the text is not a payroll action.", &payroll, text.into(), "none");
    }
    let withhold = ["August", "September", "last March", "the last pay cycle"];
    for month in withhold {
        choice(out, "abstain", "Which payroll action is requested? Choose none if the text is not a payroll action.", &payroll, format!("Please correct the tax withholding on my {month} paycheck."), "correct a tax withholding");
    }
    let legal = ["file a trademark", "send a cease and desist", "none"];
    let not_legal = [
        "The shop opens at nine.",
        "Two plus two is four.",
        "The fern by the window needs water.",
        "Standup is at 9:30 in the small room.",
        "I liked the keynote.",
        "The printer on two is out of paper.",
        "Remember to badge out.",
        "The prototype is on the table.",
    ];
    for text in not_legal {
        choice(out, "abstain", "Which legal filing does the text request? Choose none if it requests none.", &legal, text.into(), "none");
    }
}

fn routing(out: &mut Vec<Item>) {
    let q = "Which team should own this message? Choose none if none of these teams own it.";
    let opts = ["billing", "account access", "shipping", "none"];
    let months = ["April", "May", "June", "July", "August", "September"];
    let charges = ["invoice", "subscription", "order"];
    for month in months {
        for charge in charges {
            choice(out, "routing", q, &opts, format!("I was billed twice for the {month} {charge} and I want the duplicate charge returned."), "billing");
        }
    }
    let symptoms = [
        "I cannot sign in. The password reset email never arrives.",
        "My account is locked after too many attempts.",
        "The two-factor code is rejected even when I type it immediately.",
        "I never received the invitation to set a password.",
        "Single sign-on sends me back to the login page.",
        "I need the email on my account changed so I can sign in.",
    ];
    for text in symptoms {
        choice(out, "routing", q, &opts, text.into(), "account access");
    }
    let goods = ["order", "parcel", "replacement part", "return shipment", "sample kit", "trade-show crate"];
    for good in goods {
        choice(out, "routing", q, &opts, format!("The tracking page still shows my {good} sitting at the warehouse from last week."), "shipping");
    }
    let other = [
        "What time does the downtown shop close on Sundays?",
        "Do you validate parking?",
        "Is the workshop dog friendly?",
        "Which floor is reception on?",
        "Do you have a vegetarian option at the cafe?",
        "Where should visitors wait?",
    ];
    for text in other {
        choice(out, "routing", q, &opts, text.into(), "none");
    }
}

fn expense(out: &mut Vec<Item>) {
    let q = "Which expense category is this?";
    let opts = ["ground transport", "lodging", "meals", "software", "none"];
    let rides = [("airport", "client office", 46), ("hotel", "venue", 18), ("office", "train station", 22), ("site", "rental counter", 31)];
    let days = ["March 2", "April 9", "June 14"];
    for (a, b, amount) in rides {
        for day in days {
            choice(out, "expense", q, &opts, format!("Taxi from the {a} to the {b}, ${amount}, {day}."), "ground transport");
        }
    }
    let hotels = ["Harbor Hotel", "Market Inn", "Station Lodge"];
    let nights = [1, 2, 3];
    for hotel in hotels {
        for night in nights {
            choice(out, "expense", q, &opts, format!("{night} nights at the {hotel} during the offsite."), "lodging");
        }
    }
    let tools = ["design tool", "error tracker", "password manager", "video editor"];
    for tool in tools {
        choice(out, "expense", q, &opts, format!("Annual seat for the {tool}, billed to the company card."), "software");
    }
    let meals = ["team lunch", "client dinner", "interview coffee", "working breakfast"];
    let amounts = [28, 64, 96, 140];
    for meal in meals {
        for amount in amounts {
            choice(out, "expense", q, &opts, format!("{meal} after the workshop, ${amount} including tip."), "meals");
        }
    }
}

fn priority(out: &mut Vec<Item>) {
    let q = "How urgent is this for the on-call team?";
    let levels = ["low", "medium", "high", "urgent"];
    let nits = ["footer typo", "extra space in a label", "outdated copyright year", "misspelled tooltip"];
    for nit in nits {
        score(out, "priority", q, &levels, format!("A {nit} shipped in the newsletter. Purchases still succeed."), "low");
    }
    let slows = ["search", "the dashboard", "image upload", "the settings page"];
    for page in slows {
        score(out, "priority", q, &levels, format!("{page} takes about three seconds for some users, and the results are still correct."), "medium");
    }
    let regions = ["one region", "new signups in Europe", "the mobile app in Canada"];
    for region in regions {
        score(out, "priority", q, &levels, format!("{region} is erroring. Other regions are fine, and support volume is rising."), "high");
    }
    let outs = ["Checkout", "Card payments", "Login", "Order submission"];
    for system in outs {
        score(out, "priority", q, &levels, format!("{system} has failed for every customer for the last 40 minutes. Nothing is completing."), "urgent");
    }
}

fn sentiment(out: &mut Vec<Item>) {
    let q = "How positive is this message?";
    let levels = ["very negative", "negative", "neutral", "positive", "very positive"];
    let products = ["onboarding", "support team", "reporting tool", "mobile app", "billing portal"];
    for product in products {
        score(out, "sentiment", q, &levels, format!("This is the clearest {product} I have used. I had my team set up the same day."), "very positive");
        score(out, "sentiment", q, &levels, format!("The {product} is helpful. A few steps were confusing, but I finished."), "positive");
        score(out, "sentiment", q, &levels, format!("The {product} address and hours are listed on the contact page."), "neutral");
        score(out, "sentiment", q, &levels, format!("The {product} was slower than I expected and the steps were clumsy."), "negative");
        score(out, "sentiment", q, &levels, format!("The {product} never worked, nobody replied, and I was charged anyway."), "very negative");
    }
}

fn news(out: &mut Vec<Item>) {
    let q = "Which news desk should take this story?";
    let opts = ["business", "sports", "science", "arts", "politics"];
    let companies = ["a chip maker", "a grocery chain", "a regional bank", "an airline"];
    let moves = ["reported quarterly earnings", "announced a merger", "cut its dividend", "raised prices"];
    for company in companies {
        for mv in moves {
            choice(out, "news", q, &opts, format!("Today {company} {mv}."), "business");
        }
    }
    let games = ["the final", "extra time", "the home opener", "the qualifying match"];
    let sports = ["scored twice", "saved a penalty", "won in overtime", "clinched the series"];
    for game in games {
        for act in sports {
            choice(out, "news", q, &opts, format!("In {game}, the striker {act}."), "sports");
        }
    }
    let labs = ["the university lab", "the observatory", "the hospital study", "the climate group"];
    let finds = ["published a vaccine trial update", "measured a new exoplanet", "mapped a glacier's retreat", "tested a faster battery"];
    for lab in labs {
        for find in finds {
            choice(out, "news", q, &opts, format!("{lab} {find}."), "science");
        }
    }
}

fn document(out: &mut Vec<Item>) {
    let q = "What kind of document is this?";
    let opts = ["invoice", "resume", "agenda", "lease", "recipe"];
    let vendors = ["Northwind Supply", "Harbor Print", "Oak Street Catering", "Lumen Electric"];
    let amounts = [240, 890, 1500, 76];
    for vendor in vendors {
        for amount in amounts {
            choice(out, "document", q, &opts, format!("Invoice from {vendor}. Amount due ${amount}. Please pay within 30 days."), "invoice");
        }
    }
    let people = ["Mina", "Omar", "Priya", "Luis"];
    let years = [3, 6, 10, 15];
    for person in people {
        for year in years {
            choice(out, "document", q, &opts, format!("{person} has {year} years as an accountant. Skills: Excel, close, audit support. Seeking a senior role."), "resume");
        }
    }
    let meetings = ["staff meeting", "design critique", "board call", "training day"];
    for meeting in meetings {
        choice(out, "document", q, &opts, format!("Agenda for the {meeting}: introductions, review, decisions, next steps. One hour."), "agenda");
    }
    let units = ["apartment 4B", "the studio", "suite 200", "the ground-floor shop"];
    for unit in units {
        choice(out, "document", q, &opts, format!("Lease for {unit}. Term is twelve months. Rent is due on the first. The tenant may not sublet."), "lease");
    }
    let dishes = ["soup", "roast", "bread", "salad"];
    for dish in dishes {
        choice(out, "document", q, &opts, format!("Recipe for {dish}. Heat the oven, combine the ingredients, and cook until done. Serves four."), "recipe");
    }
}

fn industry(out: &mut Vec<Item>) {
    let q = "Which industry is this organization in?";
    let opts = ["healthcare", "banking", "retail", "energy", "education"];
    let hospitals = ["clinic", "hospital", "urgent care", "pediatric practice"];
    for place in hospitals {
        choice(out, "industry", q, &opts, format!("The {place} added evening appointments and hired two nurses."), "healthcare");
    }
    let banks = ["credit union", "community bank", "mortgage desk", "card issuer"];
    for bank in banks {
        choice(out, "industry", q, &opts, format!("The {bank} raised savings rates and opened a new branch."), "banking");
    }
    let shops = ["grocery chain", "clothing shop", "hardware store", "pharmacy chain"];
    for shop in shops {
        choice(out, "industry", q, &opts, format!("The {shop} reported weekend foot traffic and restocked the shelves."), "retail");
    }
    let plants = ["utility", "wind farm", "solar installer", "grid operator"];
    for plant in plants {
        choice(out, "industry", q, &opts, format!("The {plant} brought a new generating unit online before summer demand."), "energy");
    }
    let schools = ["elementary school", "community college", "training academy", "university department"];
    for school in schools {
        choice(out, "industry", q, &opts, format!("The {school} published the fall course list and opened enrollment."), "education");
    }
}

fn country(out: &mut Vec<Item>) {
    let q = "Which country is this describing?";
    let opts = ["France", "Japan", "Brazil", "Egypt", "Canada", "India"];
    let facts = [
        ("The capital is Paris and the river through it is the Seine.", "France"),
        ("The notes mention the Louvre and a train to Lyon.", "France"),
        ("The office is in Tokyo and the team took the shinkansen to Osaka.", "Japan"),
        ("The meeting is in Kyoto during cherry blossom week.", "Japan"),
        ("The port call is in Rio and the notes mention Portuguese and Carnival.", "Brazil"),
        ("The factory visit is in Sao Paulo and the brief is in Portuguese.", "Brazil"),
        ("The site visit is in Cairo, beside the Nile, with a day at Giza.", "Egypt"),
        ("The partner's office is in Luxor and the notes mention the Nile.", "Egypt"),
        ("The workshop is in Toronto and guests are crossing from Montreal.", "Canada"),
        ("The retreat is in Vancouver with a train through the Rockies.", "Canada"),
        ("The client is in Mumbai and the follow-up is in Delhi.", "India"),
        ("The offsite is in Bengaluru and the notes mention the monsoon.", "India"),
    ];
    for (text, gold) in facts {
        choice(out, "country", q, &opts, text.into(), gold);
    }
}

fn cuisine(out: &mut Vec<Item>) {
    let q = "Which cuisine is this menu describing?";
    let opts = ["Italian", "Japanese", "Mexican", "Indian", "French"];
    let lines = [
        ("fresh pasta, tomato sauce, and tiramisu", "Italian"),
        ("risotto, basil, and espresso", "Italian"),
        ("sushi, miso soup, and green tea", "Japanese"),
        ("ramen, pickled ginger, and rice", "Japanese"),
        ("tacos, salsa, and lime", "Mexican"),
        ("enchiladas, beans, and cilantro", "Mexican"),
        ("dal, naan, and cardamom", "Indian"),
        ("biryani, chutney, and cumin", "Indian"),
        ("a baguette, butter, and onion soup", "French"),
        ("croissant, gruyere, and a small salad", "French"),
    ];
    for (line, gold) in lines {
        choice(out, "cuisine", q, &opts, format!("Tonight's menu: {line}."), gold);
    }
}

fn profession(out: &mut Vec<Item>) {
    let q = "Which profession is this person practicing?";
    let opts = ["accountant", "nurse", "teacher", "plumber", "pilot"];
    let lines = [
        ("She closed the books and filed the quarterly return.", "accountant"),
        ("He reconciled the ledger and prepared the tax packet.", "accountant"),
        ("She checked vitals and updated the patient's chart.", "nurse"),
        ("He started an IV and recorded the medication.", "nurse"),
        ("She assigned homework and graded the quizzes.", "teacher"),
        ("He taught the morning class and met parents after school.", "teacher"),
        ("She replaced the leaking pipe under the sink.", "plumber"),
        ("He cleared the drain and installed a new faucet.", "plumber"),
        ("She completed the preflight check and taxied to the runway.", "pilot"),
        ("He flew the evening route and logged the hours.", "pilot"),
    ];
    for (line, gold) in lines {
        choice(out, "profession", q, &opts, line.into(), gold);
    }
}

fn meeting(out: &mut Vec<Item>) {
    let q = "What kind of meeting is this?";
    let opts = ["standup", "interview", "sales call", "retrospective", "none"];
    let teams = ["platform", "billing", "mobile", "support"];
    for team in teams {
        choice(out, "meeting", q, &opts, format!("Daily {team} standup. Yesterday, today, and blockers. Fifteen minutes."), "standup");
        choice(out, "meeting", q, &opts, format!("Interview with a candidate for the {team} role. Portfolio first, then questions."), "interview");
        choice(out, "meeting", q, &opts, format!("Call with a prospect about buying the {team} package. Pricing is on the agenda."), "sales call");
        choice(out, "meeting", q, &opts, format!("{team} retrospective. What went well, what did not, and one change for next sprint."), "retrospective");
    }
}

fn product(out: &mut Vec<Item>) {
    let q = "Which catalog is this product in?";
    let opts = ["electronics", "apparel", "grocery", "furniture"];
    let goods = [
        ("wireless headphones with a charging case", "electronics"),
        ("a 27-inch monitor and an HDMI cable", "electronics"),
        ("a wool coat, size medium", "apparel"),
        ("cotton shirts in three colors", "apparel"),
        ("a carton of milk and a loaf of bread", "grocery"),
        ("rice, beans, and olive oil", "grocery"),
        ("an oak dining table with four chairs", "furniture"),
        ("a sofa and a floor lamp", "furniture"),
    ];
    for (text, gold) in goods {
        choice(out, "product", q, &opts, format!("SKU note: {text}."), gold);
    }
}

fn weather(out: &mut Vec<Item>) {
    let q = "Which condition is the forecast describing?";
    let opts = ["rain", "snow", "heat", "wind", "fog"];
    let lines = [
        ("Umbrellas are advised. Showers continue through the evening.", "rain"),
        ("Steady rain and standing water on the roads.", "rain"),
        ("Flurries overnight and ice on the steps by morning.", "snow"),
        ("Several inches of snow and a school delay.", "snow"),
        ("Highs near 100 degrees and a heat advisory.", "heat"),
        ("Stay indoors during the hottest part of the afternoon.", "heat"),
        ("Gusts above 40 miles an hour and loose branches.", "wind"),
        ("A wind advisory is up for the ridge.", "wind"),
        ("Visibility is under a quarter mile until the sun burns it off.", "fog"),
        ("Dense morning fog is delaying flights.", "fog"),
    ];
    for (line, gold) in lines {
        choice(out, "weather", q, &opts, line.into(), gold);
    }
}

fn hr(out: &mut Vec<Item>) {
    let q = "Which personnel action is this?";
    let opts = ["offer", "promotion", "onboarding", "departure", "none"];
    let roles = ["accountant", "designer", "analyst", "manager"];
    for role in roles {
        choice(out, "hr", q, &opts, format!("We are ready to send {role} candidate Jordan a written offer with a start date."), "offer");
        choice(out, "hr", q, &opts, format!("Jordan is moving from {role} to a senior {role} title and a new salary band."), "promotion");
        choice(out, "hr", q, &opts, format!("Jordan's first day as {role}: badge, laptop, and a buddy on the team."), "onboarding");
        choice(out, "hr", q, &opts, format!("Jordan's last day as {role} is Friday. Please collect the badge and transfer the files."), "departure");
    }
}

fn compare(out: &mut Vec<Item>) {
    let pairs = [(2, 5), (10, 3), (8, 8), (100, 40), (7, 9), (15, 15), (4, 1), (20, 25)];
    for (a, b) in pairs {
        let state = format!("The first quote is {a}. The second quote is {b}.");
        noul(out, "compare", state.clone(), format!("The first quote is larger than the second."), a > b);
        noul(out, "compare", state, format!("The two quotes are equal."), a == b);
    }
}

fn risk(out: &mut Vec<Item>) {
    let q = "How serious is this risk?";
    let levels = ["low", "moderate", "high", "severe"];
    let lows = ["a typo on an internal wiki page", "a late slide in an internal draft"];
    for text in lows {
        score(out, "risk", q, &levels, format!("The only issue is {text}. Customers are unaffected."), "low");
    }
    let mods = ["a backup that succeeded but ran an hour late", "a certificate that expires in 45 days"];
    for text in mods {
        score(out, "risk", q, &levels, format!("We noticed {text}. There is time to fix it before anyone is blocked."), "moderate");
    }
    let highs = ["admin access shared by three people", "backups that have not been tested in a year"];
    for text in highs {
        score(out, "risk", q, &levels, format!("The review flagged {text}. A failure would be hard to recover from quickly."), "high");
    }
    let severes = ["a public storage bucket with customer records", "production credentials committed in a public repository"];
    for text in severes {
        score(out, "risk", q, &levels, format!("Right now there is {text}. Outsiders can already reach the data."), "severe");
    }
}

fn folder(out: &mut Vec<Item>) {
    let q = "Which mailbox folder should this message go in?";
    let opts = ["work", "newsletter", "personal", "suspicious", "none"];
    let work = [
        "The client moved the review to 3pm. Please update the agenda.",
        "Finance needs the March accrual before close.",
        "The deploy window is Thursday at 6pm.",
        "Legal redlined the statement of work.",
    ];
    for text in work {
        choice(out, "folder", q, &opts, text.into(), "work");
    }
    let news = [
        "This week's product roundup is inside. Unsubscribe at the bottom.",
        "Your monthly digest of industry headlines.",
        "New posts from the blogs you follow.",
        "Sale ends Sunday. View this email in a browser.",
    ];
    for text in news {
        choice(out, "folder", q, &opts, text.into(), "newsletter");
    }
    let personal = [
        "Mom called. Dinner is at 6 on Sunday.",
        "Can you pick up the kids after practice?",
        "The dentist confirmed your cleaning on Tuesday.",
        "Happy birthday. Cake is in the fridge.",
    ];
    for text in personal {
        choice(out, "folder", q, &opts, text.into(), "personal");
    }
    let suspicious = [
        "Your account will close in one hour unless you enter your password at this unfamiliar link.",
        "We detected a login. Send us your one-time code to confirm you are the owner.",
        "Payroll direct deposit failed. Reply with your bank login.",
        "An invoice is attached. Enable macros and type your password to view it.",
    ];
    for text in suspicious {
        choice(out, "folder", q, &opts, text.into(), "suspicious");
    }
}
