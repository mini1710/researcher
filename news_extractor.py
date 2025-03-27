

from dotenv import load_dotenv
import os
import configparser
import json
import re
import numpy as np
from rapidfuzz import fuzz
from datetime import datetime
import logging

from typing import Optional,List,Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from tavily import TavilyClient
import warnings
warnings.filterwarnings("ignore")


def create_logger():
  logger = logging.getLogger('logger')
  logger.setLevel(logging.INFO)

  file_handler = logging.FileHandler('app.log',mode="w")
  file_handler.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  logger.info("Logging initialized!")
  return logger,file_handler


def stop_logger(logger, file_handler):
    file_handler.close() 
    logger.removeHandler(file_handler)




load_dotenv()

def load_properties(file_path="config.conf"):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config 

config = load_properties()

model = config.get("GENERAL", "model", fallback="gpt-4o-mini")
temperature = config.getint("GENERAL", "temperature", fallback=0)
research_domains = config.get("SEARCH", "research_domains").split(", ")
search_query = config.get("SEARCH", "search_query")
search_limit = config.getint("SEARCH", "search_limit", fallback=5)
search_depth = config.get("SEARCH", "search_depth")
news_instructions = config.get("SEARCH","news_instructions")
search_research_instructions = config.get("VALIDATE", "search_research_instructions")
extract_instructions = config.get("VALIDATE", "extract_instructions")
instruction_research = config.get("EXTRACT", "instruction_research")


class SearchState(TypedDict):
  query: Annotated[str,"The query for the search client"]
  is_news: Annotated[bool,"The query type News(True) or General(False)"]
  time_range: Annotated[str,"The duration for fetching the news"]
  include_domains : Annotated[List[str],"Which news portals or research portals to look into"]
  news_results: Annotated[List[dict], "the extracted news"]
  research_results: Annotated[List[dict],"extracted results"]
  failed_extract: Annotated[List[str], 'urls of broken /unavailable pages']
  urls: Annotated[List[str], "valid research urls"]

class NewsItem(BaseModel):
  url : str =  Field(description="The source of the news article")
  title : str = Field(description="The title of the news article")
  source: Optional[str] = Field(default=None,description = "The news agency that published this news")
  content: str = Field(description="A summary of the news article")
  published_date : str = Field(description="The publication date of the news article")
  tags: Optional[List[str]] = Field(default_factory=list, description="A list of tags associated with the news article")

class News(BaseModel):
  news_results : list[NewsItem] = Field(description="A list of news articles")

class ResearchItem(BaseModel):
  url : str =  Field(description="The source of the research article")
  title : str = Field(description="The title of the research article")
  abstract : str = Field(description="The abstract of the research article")
  authors : str =  Field(description="The authors of the research article")
  date : str = Field(description="The publication date of the research article")
  tags : Optional[List[str]] = Field(default_factory=list, description="A list of tags associated with the research article")

class Research(BaseModel):
  research_results : list[ResearchItem] = Field(description="A list of Research articles")

class URL(BaseModel):
    urls: List[str] = Field(description="List of research article urls")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")



search_client = TavilyClient(TAVILY_API_KEY)
news_parser = PydanticOutputParser(pydantic_object=News)
research_parser = PydanticOutputParser(pydantic_object=Research)
url_parser = PydanticOutputParser(pydantic_object=URL)
llm = ChatOpenAI(model_name=model, temperature=temperature,openai_api_key=OPENAI_API_KEY)

logger,fh = create_logger()
fh.flush()

def check_string_for_404_or_page_not_found(input_string):
    pattern = r"(404|page\s*not\s*f[o0]und)"
    if re.search(pattern, input_string, re.IGNORECASE):
        return True
    return False

def remove_duplicate_titles(news_items: list[NewsItem]):
  seen_titles = set()
  unique_news = []
  for item in news_items:
    if item['title'] not in seen_titles:
      if not seen_titles:
        seen_titles.add(item['title'])
        unique_news.append(item)
      else:
        max_match = int(np.max([(fuzz.ratio(item['title'],x)) for x in seen_titles]))
        if max_match<95:
            seen_titles.add(item['title'])
            unique_news.append(item)
  return unique_news

def search_web(state):
  try:
    response = search_client.search(query=state['query'],
                      topic = "news" if state['is_news'] else "general",
                      time_range=state['time_range'],
                      max_results=search_limit,
                      include_answer="basic",
                      include_domains=[] if state['is_news'] else research_domains,
                      exclude_domains=["propertycasualty360.com",'X.com'],
                      include_raw_content=False)
    if state['is_news']:
      state['news_results'] = response.get('results')
      if len(state['news_results'])>1:
        news_items = remove_duplicate_titles(state['news_results'])
        parsed_news = News(news_results = news_items)
      else:
        parsed_news = News(news_results = state['news_results'])
      state['news_results'] = []
      for news in parsed_news.news_results:
        news_prompt = PromptTemplate(
            input_variables=["news_items"],
            template=news_instructions,
            partial_variables={"news_format_instructions" : news_parser.get_format_instructions()}
            )
        formatted_prompt = news_prompt.format(news_items=news)
        output = llm.invoke(formatted_prompt)
        output_parsed = news_parser.parse(output.content)
        filtered_news_results = [
        news_item for news_item in output_parsed.news_results 
        if news_item.tags not in [[], [None]]]
        output_parsed.news_results = filtered_news_results
        state['news_results'].extend(output_parsed.news_results)
    else:
      state['research_results'] = response.get('results')
  except:
    logger.error("Search failed")
  return state

def extract_tavily(state):
  response_list= []
  if len(state['urls'])>0:
    for url in state['urls']:
      try:
        response = search_client.extract(url)
        if response['failed_results']!= []:
          state['failed_extract'].extend(response['results'])
        else:
          response_list.append(response)
      except:
        logger.error(f"Error extracting {url}")
        continue
  else:
    response_list = []
    logger.info("No Results Found")
  return response_list



def validate_news(state):
  search_template = PromptTemplate(
      input_variables=["news"],
      template=search_research_instructions.strip(),
  )

  extract_template = PromptTemplate(
      input_variables=["content","news"],
      template= extract_instructions,
      partial_variables={"url_format_instructions": url_parser.get_format_instructions()},
      )
  
  state['query'],state['urls'] = [],[]
  state['is_news'],state["include_domains"],state['time_range'] = False,research_domains,"y"
  
  for news in state['news_results']:
    formatted_research = search_template.format(news=news.content)
    try:
      search_query = llm.invoke(formatted_research)
      state['query'] = search_query.content
      
      response = search_web(state)
      formatted_pr = extract_template.format(search_results=response['research_results'],news=news)
      url_list = llm.invoke(formatted_pr)
      state["urls"].extend(url_parser.parse(url_list.content).urls)
    except:
      logger.error("LLM invocation failed, URLs not extracted")
  return state



def run_extraction(state):
  extracted_papers = extract_tavily(state)
  state['research_results'] =[]
  if extracted_papers not in [[], [None], "null", ["null"], None]:
    research_prompt = PromptTemplate(
            template=instruction_research,
            input_variables=["extracted_papers"],
            partial_variables={"research_format_instructions": research_parser.get_format_instructions()},
            )
    formatted_prompt_research = research_prompt.format(extracted_papers=extracted_papers)
    try:
      output =llm.invoke(formatted_prompt_research)
      parsed_output = research_parser.parse(output.content)
      filtered_research_results = [
      row for row in parsed_output.research_results
      if (row.tags not in [[], [None], "null", ["null"], None]) &
      (check_string_for_404_or_page_not_found(row.abstract) == False)]

      parsed_output.research_results = filtered_research_results
      state['research_results'].extend(parsed_output.research_results)
    except:
      logger.error("LLM invocation failed")
  return state

def save_results_to_file(data):
  try:
    news_results = News(news_results=data['news_results'])
    research_results = Research(research_results=data['research_results'])
    if isinstance(news_results, News):
      news_dict = News.model_dump(news_results)
      if isinstance(research_results, Research):
        research_dict = Research.model_dump(research_results) 
      database_dict = {"news_dict": news_dict, "research_dict": research_dict}

    else:
      database_dict = {"news_dict": {
                        "news_results":
                            {
                                "url": "",
                                "source": [],
                                "title": "",
                                "content": "",
                                "published_date": "",
                                "tags": []
                            }
                    },
                    "research_dict": {
                        "research_results":
                            {
                                "url": "",
                                "title": "",
                                "abstract": "",
                                "authors": "",
                                "date": "",
                                "tags": []
                            }}}
    timestamp = datetime.now().strftime("%d %b %y , %a  %H:%M")
    database_dict['last_refreshed_date'] = timestamp
    try:
      with open("database.json", "w", encoding="utf-8") as f:
        json.dump(database_dict, f, ensure_ascii=False, indent=4)
      logger.info("Saved to database.json")

    except:
      logger.error("Error with file write : Unable to save to database.json")
    
  except:
    logger.error("Error with data parse : Unable to save to file database.json")
  
 
def search_extract_news(state):
    try:
      workflow = StateGraph(tuple(state))
      workflow.add_node("search",search_web)
      workflow.add_node("invoke_yoda",validate_news)
      workflow.add_node("extract_knowledge",run_extraction)
      workflow.add_edge("search","invoke_yoda")
      workflow.add_edge("invoke_yoda", "extract_knowledge")
      workflow.add_edge("extract_knowledge",END)
      workflow.set_entry_point("search")  
      executor = workflow.compile()
      results = executor.invoke(state)
      save_results_to_file(results)
      logger.info("Workflow execution completed")
      return True
    except Exception as e:
      logger.error(f"Unable to execute graph : {e}")
      return False
    finally:
      stop_logger(logger,fh)

