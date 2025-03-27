import streamlit as st
import json
import time
import news_extractor

if "refresh" not in st.session_state:
    st.session_state.refresh = False


def capitalize_abbreviations(tags):
    taglist = tags.split()
    ignore_list = ['in','of','to','the']
    capitalized_words = [tag.upper() if (len(tag) < 4) and (tag not in ignore_list) else tag.capitalize() for tag in taglist ]
    return ' '.join(capitalized_words)

def filter_by_tag(data, selected_tag):
    return [item for item in data if any(selected_tag.lower() == tag.lower() for tag in item["tags"])]

def get_all_tags(dataset):
    tags = set()
    for label in dataset:
         for item in label:
            if item =="tags":
                formatted_tags = [capitalize_abbreviations(tag) for tag in label[item]]
                tags.update(formatted_tags)
            else:
                continue
    return sorted(tags)

def select_common_tags(news,research):
    tags_news = get_all_tags(news)
    tags_research = get_all_tags(research)
    tags_all = list(set(tags_news).union(set(tags_research)))
    return tags_all

def get_last_refreshed(dataset):
    return dataset['last_refreshed_date']

def display_error(msg,seconds):
    st.toast(msg)
    time.sleep(seconds)

with open("database.json", "r", encoding="utf-8") as file:
    db = json.load(file)

refreshed_time = get_last_refreshed(db)
news_articles = {}
research_papers = {}
search_query = news_extractor.search_query.strip()
graphstate = news_extractor.SearchState(query=search_query,
                                        time_range="month",
                                        is_news=True,
                                        include_domains=[],
                                        failed_extract= [])

def invoke_workflow():
    with st.spinner("Refreshing Data ..."):
        run = news_extractor.search_extract_news(graphstate)
    if run == False:
        display_error("Workflow failed! Check app.log for error details",2)
        st.session_state.refresh = True


if st.session_state.refresh:
    st.session_state.refresh = False
    st.rerun()

container_action = st.container(border=False)
with container_action: 
    col1,col2,col3 = st.columns([4,4,2],gap="small", vertical_alignment="center")
    with col2:
        st.text(f"""Last Refreshed - {refreshed_time}""") 
    with col3:
        st.button(label = "",type="secondary",icon = ":material/refresh:",on_click=invoke_workflow)

st.title("Climate Risk Insurance Dashboard")
st.sidebar.header("Filter by Tag")

news_articles = db['news_dict']['news_results']
research_papers = db['research_dict']['research_results']


all_tags = select_common_tags(news_articles,research_papers)
default_index = next((i for i, tag in enumerate(all_tags) if "Climate Change" in tag), 0)
selected_tag = st.sidebar.selectbox("Select a tag", all_tags,index=default_index)

st.subheader(f"News Articles Tagged - :blue[{selected_tag}]",divider="grey")
filtered_news = filter_by_tag(news_articles, selected_tag)
if filtered_news not in [[],[None],None]:
    for article in filtered_news:
        st.subheader(article["title"])
        st.caption(f"{article['source']} | {article['published_date']}")
        st.write(article["content"])
else:
    st.write("No news articles match this tag.")


st.subheader(f"Research Papers Tagged - :blue[{selected_tag}]",divider="grey")
filtered_research = filter_by_tag(research_papers, selected_tag)
if filtered_research not in [[],[None],None]:
    for paper in filtered_research:
        st.subheader(paper["title"])
        st.caption(f"{paper['authors']} | {paper['date']}")
        st.write(paper["abstract"])
        st.caption(f"{paper['url']}")
else:
    st.write("No research papers match this tag.")

st.sidebar.markdown(
            """
            ### About
            <p style="font-size: 13px;">
            This dashboard integrates news and academic research on climate risk insurance.
            Select a tag to see related items from both News and arXiv research
            </p>

            """,
            unsafe_allow_html=True
        )

