# StefanosTsitsipADA: REDDIT MODERATOR'S SURVIVAL GUIDE

> [!NOTE]
> URL: https://xarsevandor.github.io/ADA-Project/

## Abstract
This research investigates the multidimensional nature of intercommunity hostility on Reddit using the SNAP Reddit Hyperlink Network dataset (2014-2017). By moving beyond raw interaction volumes, we identify structural aggressors through toxicity ratios and categorize subreddits into tactical roles such as "Snipers," "Victims," and "War Zones." Utilizing UMAP dimensionality reduction on user navigation patterns, we define 19 distinct community "Tribes" to demonstrate that conflict is primarily driven by semantic proximity where communities attack "neighbors" with small ideological differences rather than random strangers.

Our core temporal analysis focuses on analyzing where the conflict is by detecting cascades events where aggression against one community triggers a subsequent, contagious strike against a third party. We further extract the linguistic signatures and sentiment shifts of these hostile events using VADER sentiment analysis and perform forensic audits on the temporal "flatlines" of communities leading up to their eventual ban. This framework serves as a comprehensive diagnostic tool for moderators to identify, track, and predict the propagation of toxicity across the Reddit ecosystem.

## Research Questions
1. **Topological Intelligence**
    - How can ratio-based metrics distinguish between active "public squares" and toxic aggressors?
	- What structural roles (Snipers, Victims, War Zones) do communities play in a conflict network?
2.	**Semantic Proximity**
	- Is conflict driven by ideological friction between similar communities?
	- How does distance correlate with the volume and intensity of hostility?
3.	**Conflict Cascades**
	- Can we statistically identify "cascades" where hostility propagates through the network?
	- How do hubs" like r/SubredditDrama signal these events?
4.	**Linguistic Signatures**
    - What specific linguistic markers, sentiment shifts, and readability changes characterize hostile cross-community interactions?

## Proposed Additional Datasets
We will focus on the two main datasets that are proposed and combine them with the embeddings dataset. 

## Project Plans & Methods
### Data Processing
We began by exploring and cleaning the Subreddit Hyperlink Network datasets as well the Subreddit Embeddings datasets. We had to merge the POST_PROPERTIES metadata, parse the timestamps, and handle missing values. We also merged for some analysis title-based and body-based interaction datasets to get a general overview.

### Part I: Know Your Enemy, Topological Anlyisis
The first phase of our analysis focuses on the structural topology of the Reddit network. By examining the ratio of outgoing to incoming negative links a metric we call the "False Friend Paradox" we distinguish between high traffic hubs and truly aggressive actors. Communities are categorized into tactical roles such as "Snipers" (high aggressors), "Victims" (frequent targets), and "War Zones" (reciprocal conflict). This allows for an understanding of the intent behind community interactions, moving beyond simple volume metrics to identify the actual sources of instability within the network.

### Part II: The Semantic Radar: Motivation Analysis
To understand the motivations behind these attacks, we map the ideological landscape of Reddit using UMAP dimensionality reduction on user navigation patterns. This coordinate system clusters subreddits into 19 distinct "Tribes," such as politics, gaming, or sports. Our findings demonstrate the "Law of Proximity," revealing that conflict is rarely random; instead, it most frequently occurs between topically similar communities where ideological friction is highest. By quantifying the distance between these clusters, we prove that subreddits tend to fight their "neighbors" rather than distant strangers.

### Part III: Signal Intelligence
The core of our temporal investigation focuses on analyzing where the conflict is by tracking its propagation through the network over time. We specifically search for "Conflict Cascades," defined as events where an attack on one subreddit triggers a subsequent, non-retaliatory strike by the victim against an unrelated third party target. By identifying these chains and monitoring "Signal Intelligence" hubs like r/SubredditDrama, we can trace how hostility migrates across the platform. This analysis allows us to pinpoint the active locations of warfare and identify the subreddits that act as carriers for spreading conflict.

### Part IV: The Language of War: Sarcasm and Linguistic Analysis
This module of our survival guide addresses the complex linguistic signatures of inter community hostility, moving beyond the rudimentary approach of simple filtering. We demonstrate that identifying conflict requires more than just a list of "bad words," as hostile intent on Reddit is frequently masked by sarcasm. By utilizing LIWC (Linguistic Inquiry and Word Count) feature analysis, we examine the deeper psychological and structural markers of the text. This analysis allows us to detect "hidden" toxicity where the surface level sentiment might appear neutral or humorous, but the underlying linguistic features signal a coordinated strike. Through this lens, we also conduct forensic audits of "Cold Case Files," tracing the linguistic evolution and terminal activity patterns of subreddits leading up to their eventual ban by platform administrators.

### Part V: The War Room (Spatio-Temporal Analysis)
The culminating module of our research focuses on When & Where conflicts ignite by synchronizing spatial coordinates with temporal spikes. By mapping the "Tidal Waves" of hostility, we analyze how conflict intensity fluctuates across different time zones and subcommunities simultaneously. This allows us to move from observing static relationships to identifying the "Peak Hours of Hostility" and the specific network locations that act as global pressure points. Using interactive spatio-temporal visualizations, we provide moderators with a "War Room" perspective, enabling them to see not just which communities are at risk, but exactly when the network is most vulnerable to synchronized attacks or organic escalations.

## Proposed Timeline
| Date       | Milestone                                                            |
|------------|----------------------------------------------------------------------|
| 10.11.2025 | First thorough analysis                                              |
| 20.11.2025 | Refinment based on feedback                                          |
| 22.11.2025 | Homework                                                             |
| 30.12.2025 | Final analysis and visualization results                             |
| 03.12.2025 | Handling network data                                                |
| 07.12.2025 | Begining of Website                                                  |
| 17.12.2025 | Final review of the website and cleaning of the repository           |

## Organization within the team

- **Tomas** Part 1 & 5
- **Alexandre** Part 2
- **Gaspard** Part 3
- **Rodrigue** Part 4
- **Georgios** Website

## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>

# install requirements
pip install -r pip_requirements.txt

# Download the data set(s)
cd scripts
chmod +x load_data.sh
./load_data.sh
```

### How to use the library
To load the data from the main notebook or from another `.py` file, run:
```py
from data_processing.load_data import load_data
full_title_df, full_body_df = load_data()
```
> [!IMPORTANT]
> Do not commit the pickle files on github

To display some plots interactively (to see the evolution through time) set the `snapshot` variable at the top of [results.ipynb](results.ipynb) to `None`.\
If you only want to display a snapshot of the plot (i.e. to push on GitHub), set it to a specific month.
```py
snapshot=None
# or
snapshot="2016-11"
```

## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── final.ipynb               <- a well-structured notebook showing the results
│
├── index.html                <- main html file for the website
├── liwc_radar.html
├── top_links_with_subreddit_time_slider.html
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

