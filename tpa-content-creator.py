# for streamlit code
import streamlit as st
from config import config
import requests
import pandas as pd
from collections import Counter
from urllib.parse import quote
import logging
from config import config
import aisuite as ai
import time
import os

# --- FUNCTIONS.PY ---
def handle_api_errors(response, api_name):
    if response.status_code != 200:
        logging.error(f"Failed to retrieve data from {api_name}. Status code: {response.status_code}")
        return False
    return True

# --- Step 2: SerpAPI Data Retrieval ---
def get_serpapi_data(topic_query, SERPAPI_KEY):
    base_url = "https://serpapi.com/search.json"
    params = {
        "q": topic_query,
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": SERPAPI_KEY
    }
    response = requests.get(base_url, params=params)
    if handle_api_errors(response, "SerpAPI"):
        results = response.json()
        data = []
        for result in results.get('organic_results', []):
            data.append({
                'Position': result.get('position'),
                'Link': result.get('link'),
                'Title': result.get('title')
            })
        df_serp = pd.DataFrame(data)
        logging.info("Successfully retrieved SerpAPI data.")
        return df_serp
    else:
        error_message = response.json().get('error', 'Unknown error occurred')
        logging.error(f"Error retrieving SerpAPI data: {error_message}")
        return error_message

# --- Step 3: SEMRush Data Retrieval and Processing ---
def get_semrush_data(url, api_key):
    if not api_key:
        logging.error("SEMRush API key is missing")
        return []
        
    base_url = "https://api.semrush.com/"
    type_param = "url_organic"
    export_columns = "Ph,Po,Nq,Cp,Co"
    
    try:
        full_url = (
            f"{base_url}?type={type_param}&key={api_key}"
            f"&display_limit={config['semrush_display_limit']}&export_columns={export_columns}"
            f"&url={quote(url)}&database={config['semrush_database']}"
            f"&display_filter={config['semrush_display_filter']}&display_sort={config['semrush_display_sort']}"
        )
        response = requests.get(full_url)
        
        if response.status_code == 403:
            logging.error("SEMRush API authentication failed. Please check your API key.")
            return []
        elif not handle_api_errors(response, "SEMRush"):
            return []
            
        decoded_output = response.content.decode('utf-8')
        lines = decoded_output.split('\r\n')
        headers = lines[0].split(';')
        json_data = []
        for line in lines[1:]:
            if line:
                values = line.split(';')
                record = {header: value for header, value in zip(headers, values)}
                json_data.append(record)
        return json_data
        
    except Exception as e:
        logging.error(f"Error in SEMRush API call: {str(e)}")
        return []

def process_semrush_data(df, api_key):
    df['SEMRush_Data'] = df['Link'].apply(lambda x: get_semrush_data(x, api_key))
    logging.info("Successfully retrieved SEMRush data.")
    all_keywords = []
    for data in df['SEMRush_Data']:
        if data:
            all_keywords.extend([item['Keyword'] for item in data])

    keyword_counts = Counter(all_keywords)
    highest_count = max(keyword_counts.values())
    second_highest_count = sorted(set(keyword_counts.values()), reverse=True)[1] if len(set(keyword_counts.values())) > 1 else 0
    top_keywords = [keyword for keyword, count in keyword_counts.items() if count == highest_count or count == second_highest_count]

    if highest_count == 2:
        top_keywords = [keyword for keyword, count in keyword_counts.items() if count in [1, 2]]

    search_volume_keywords = sorted(
        [(item['Keyword'], int(item['Search Volume'])) for data in df['SEMRush_Data'] if data for item in data],
        key=lambda x: x[1],
        reverse=True)[:10]

    final_keywords = set(top_keywords + [keyword for keyword, _ in search_volume_keywords])
    final_keywords_df = pd.DataFrame(
        [(keyword, 
        next((item['Search Volume'] for data in df['SEMRush_Data'] if data for item in data if item['Keyword'] == keyword), 0),
        keyword_counts[keyword])
        for keyword in final_keywords],
        columns=['Keyword', 'Search Volume', 'Frequency']
    )
    final_keywords_df = final_keywords_df.sort_values(by=['Frequency', 'Search Volume'], ascending=[False, False])

    logging.info("Successfully processed SEMRush data and extracted keywords.")
    return final_keywords_df

# --- Step 4: Content Fetching ---
def fetch_content(url, jina_api_key):
    headers = {
        'Authorization': f'Bearer {jina_api_key}',
        'X-Retain-Images': 'none',
        "Accept": "application/json",
        'X-Timeout': str(config["jina_api_timeout"])
    }
    try:
        response = requests.get(f'https://r.jina.ai/{url}', headers=headers)
        if handle_api_errors(response, "Jina AI Reader"):
            response_json = response.json()
            if response_json['code'] == 200:
                logging.info(f"Successfully fetched content from {url}.")
                return response_json['data']['content']
            else:
                logging.warning(f"Jina API error for {url}: {response_json.get('error', 'Unknown error')}")
                return f"ERROR: {url} blocks Jina API or other error occurred."
        else:
            return f"ERROR: Failed to use Jina API for {url}."

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error fetching content from {url}: {e}")
        return f"ERROR: Request failed for {url}."
    except Exception as e:
        logging.error(f"Unknown error fetching content from {url}: {e}")
        return f"ERROR: Unknown error processing {url}."

# --- Step 5-10: AI Model Interactions ---  
def interact_with_ai(messages, client, model=config["openai_model"], temperature=config["openai_temperature"]):
    try:
        response = client.chat.completions.create(
            model=model, 
            messages=messages, 
            temperature=temperature,
            )
        result = response.choices[0].message.content
        return result
    except Exception as e:
        logging.error(f"Error during AI interaction: {e}")
        return None
    
# --- CONTENT_GENERATION.PY---
# Analyze the content from top-ranking pages.
def generate_content_analysis(topic_query, df_results, client):
    messages_content_analysis = [
        {
            "role": "system", 
            "content": """You are a meticulous content researcher with expertise in analyzing web content, particularly articles and blogs. You have access to a list of webpage contents related to the topic a user is interested in."""
        },
        {
            "role": "user", 
            "content": f"""Analyze the provided content below. 

        First, determine if each piece of content is a blog or an article. Exclude any content that is primarily promotional, youtube page, a forum post, or social media content. Focus on articles and blogs that provide informative or opinionated content.

        For each identified blog or article, add it to a review list. Then, thoroughly review each item on this list and provide an analysis that includes:

        (1) Common Topics:
            * Identify 3-5 main topics discussed across the content.
            * For each main topic, list 2-3 key subtopics or related concepts.
            * Provide a brief summary (1-2 sentences) of how each subtopic is addressed.

        (2) Contradicting Viewpoints:
            * Highlight contradicting viewpoints among the top results if there are any. 
            * Briefly explain the opposing arguments and any evidence presented.

        (3) Information Gaps:
            * Considering the user's search query '{topic_query}', identify key questions that a user might have that are NOT answered by the provided content, if there are any.
            * Suggest specific areas or subtopics that are missing from the content but would be relevant to someone interested in '{topic_query}, if there are any'.

        --------------------
        """ + "\n".join([f"SOURCE {i + 1}:\n{content}" for i, content in enumerate(df_results['Content']) if content])
        }
    ]
    return interact_with_ai(messages_content_analysis, client)

# Create a detailed content plan based on analysis.
def generate_content_plan(topic_query, targeting_keywords, content_analysis, client):
    messages_content_plan = [
        {
            "role": "system", 
            "content": """You are an expert content strategist skilled in crafting detailed and actionable content plans. You are adept at creating outlines that are clear, comprehensive, and tailored to the specific needs of a given topic. You have access to a detailed analysis of competitor content related to the topic a user is interested in."""
        },
        {
            "role": "user", 
            "content": f"""Develop a comprehensive content plan considering the content analysis provided below.

        Topic: 
        {topic_query}
        
        Target Keywords:
        {targeting_keywords}
        
        Content Analysis:
        {content_analysis}
        
        The content plan should include:

        (1) Outline: 
            * Create an outline with a hierarchical structure of headings and subheadings that logically organize the content.
            * The outline should have at least 3 main headings, with subheadings under each.

        (2) SEO Keywords:
            * Incorporate these SEO keywords: {targeting_keywords}. Prioritize them with Frequency (how many top ranking pages rank on the keyword) and Search Volume 
            * Ensure these keywords are naturally integrated into the headings and subheadings where relevant.

        (3) Content Strategy:
            * Address the common topics and subtopics identified in the content analysis.
            * Highlight any areas with contradicting viewpoints, and suggest a balanced approach to these topics.
            * Fill the information gaps identified in the analysis.
            * Suggest a unique angle or perspective for the content.
            """}
    ]
    return interact_with_ai(messages_content_plan, client)

def generate_content_draft(content_plan, content_analysis, client):
    """Generate the first draft of the content."""
    messages_content_draft = [
        {"role": "system", "content": """You are a skilled content writer specializing in crafting engaging, informative, and SEO-friendly blog posts. You excel at following detailed content plans and adapting your writing style to meet specific guidelines and objectives. You have access to a content plan and an analysis of competitor content related to the topic a user is interested in."""},
        {"role": "user", "content": f"""Write a comprehensive article using the provided Content Plan and the insights from the Competitor Content Analysis:
        
        Content Plan:
        {content_plan}
        
        Content Analysis:
        {content_analysis}
        
        Requirements:
        1. Follow the provided structure exactly
        2. Maintain a natural, engaging writing style
        3. Include relevant examples and explanations
        4. Integrate target keywords naturally
        5. Focus on providing unique value"""}
    ]
    return interact_with_ai(messages_content_draft, client)

def proofread_content(content_draft, content_plan, content_analysis, client):
    """Proofread and improve the content draft."""
    messages_proofread = [
        {"role": "system", "content": """You are an expert content editor with a keen eye for detail, specializing in refining and polishing written content. You excel at ensuring content is engaging, error-free, and adheres to SEO best practices. You have access to a draft article, its corresponding content plan, and an analysis of competitor content related to the topic a user is interested in."""},
        {"role": "user", "content": f"""Review and refine the provided Content Draft to ensure it aligns with the Content Plan and surpasses the quality of competitor content.
        
        Draft:
        {content_draft}
        
        Content Plan:
        {content_plan}

        Competitor Content Analysis:
        {content_analysis}
        
        Focus on the following:
        * Accuracy: Verify the accuracy of facts and claims (if possible).
        * Clarity and Conciseness: Ensure the writing is clear, concise, and free of jargon.
        * Engagement: Suggest ways to make the content more engaging (e.g., examples, anecdotes, varied sentence structures).
        * Flow and Structure: Ensure smooth transitions between paragraphs and sections.
        * Grammar and Mechanics: Correct any grammatical errors, typos, and punctuation issues.
        * SEO:
            * Ensure headings are properly used and optimized.
            * Check for keyword density and placement (avoid keyword stuffing).

        Please provide the revised article only, without any additional commentary or explanations. """}
    ]
    return interact_with_ai(messages_proofread, client)

def generate_seo_recommendations(proofread_draft, targeting_keywords, client):
    """Generate SEO optimization recommendations."""
    messages_seo = [
        {"role": "system", "content": """You are a seasoned SEO expert specializing in optimizing blog articles for search engines. You are adept at crafting compelling title tags and meta descriptions that improve click-through rates and accurately reflect the content. You have access to the final version of a blog article and a list of its targeting keywords related to the topic a user is interested in."""},
        {"role": "user", "content": f"""Develop SEO recommendations for the provided content.
        
        Content:
        {proofread_draft}
        
        Target Keywords (Prioritized by Frequency then Search Volume):
        {targeting_keywords}
        
        URL Slug:
        * Create an optimized URL slug for the article.
        * Use hyphens to separate words.
        * Include relevant keywords.

        Title Tag:
        * Generate three variations of a Title Tag following SEO best practices.
        * Accurately reflect the content of the article.
        * Include relevant keywords (naturally and if possible).

        Meta Description:
        * Create three variations of a Meta Description.
        * Provide a concise and accurate summary of the article's content.
        * Include a call to action (if relevant).
        * Use relevant keywords (naturally).

        Please provide only the URL slug, Title Tags, and Meta Descriptions, without any additional commentary or explanations.
        """}
    ]
    return interact_with_ai(messages_seo, client)

def generate_final_deliverable(proofread_draft, seo_recommendations, targeting_keywords, df_serp, content_analysis, client):
    """Compile the final content package with all components."""
    messages_final = [
        {"role": "system", "content": """You are a meticulous Senior Project Manager with expertise in presenting comprehensive project deliverables. You excel at organizing and summarizing complex information into a clear, concise, and client-ready format. You have access to all the outputs generated during a content creation process related to the topic a user is interested in."""},
        {"role": "user", "content": f"""Compile the following information into a well-structured Markdown document for client presentation.
        
        Content:
        {proofread_draft}
        
        SEO Recommendations:
        {seo_recommendations}
        
        Target Keywords:
        {targeting_keywords}

        Competitors:
        {df_serp[['Position', 'Link', 'Title']]} 
        
        Competitor Analysis:
        {content_analysis}
        
        The Markdown document should include:

        # [Content Title]

        ## SEO Recommendations

        ### Title & Meta Description
        * Present the SEO-optimized title and meta description options.
        * Highlight the chosen or recommended ones.
        * Include alternative options for consideration.
            * Use Markdown formatting for lists and emphasis (e.g., bold, italics).

        ### URL
        * Provide the finalized URL slug for the article.

        ## Keywords Analysis
        * List the primary keywords targeted in the content.
            * Use a Markdown table to present 3 columns - keywords, search volume, and frequency.
            * Show up to 15 keywords, sorted descendingly by Frequency then Search Volume.

        ## Competitors
        * Summarize key information about the top competitors.
            * Use a Markdown table with columns for "Position", "Link", and "Title".

        ## Content Strategy Notes
        * Offer insights into the content strategy.
        * Explain what aspects are covered.
        * Highlight unique points not addressed by competitors.
        * Mention areas that may require human validation or review.

        ## ===Final Content===
        * Present the fully proofread and polished article in a markdown format

        Ensure the deliverable is client-friendly, easy to understand, and provides a comprehensive overview of the project. Please provide the final deliverable document only, without any additional commentary or explanations. 
        """}
    ]
    return interact_with_ai(messages_final, client)

# --- Initialization ---
client = ai.Client()

def sidebar_config():
    st.sidebar.title("Configuration")
    
    # API Keys Section
    st.sidebar.header("API Keys")
    api_keys = {
        'SERPAPI_KEY': st.sidebar.text_input("SerpAPI Key", type="password"),
        'SEMRUSH_API_KEY': st.sidebar.text_input("SEMrush API Key", type="password"),
        'OPENAI_API_KEY': st.sidebar.text_input("OpenAI API Key", type="password"),
        'JINA_API_KEY': st.sidebar.text_input("Jina API Key", type="password")
    }
    
    # Model Selection
    st.sidebar.header("Model Settings")
    model = st.sidebar.selectbox(
        "Select OpenAI Model",
        options=["openai:gpt-4o-mini", "openai:gpt-4o"],
        index=0
    )
    
    # Database Selection
    st.sidebar.header("SEMrush Settings")
    database = st.sidebar.selectbox(
        "Select SEMrush Database",
        options=["us", "au", "nz"],
        index=0
    )
    
    return api_keys, model, database

# def load_api_keys():
#     st.session_state.SEMRUSH_API_KEY = st.secrets['SEMRUSH_API_KEY']
#     st.session_state.SERPAPI_KEY = st.secrets['SERPAPI_KEY']
#     st.session_state.OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
#     st.session_state.JINA_API_KEY = st.secrets['JINA_API_KEY']

def main():
    st.title("TPA Content Research & Generation Agent")
    
    # Get configuration from sidebar
    api_keys, model, database = sidebar_config()

    # Update config
    config["openai_model"] = model
    config["semrush_database"] = database

    
    for key, value in api_keys.items():
        st.session_state[key] = value
    
    os.environ['OPENAI_API_KEY'] = st.session_state.OPENAI_API_KEY

    
    # Step 1: Topic Selection
    topic_query = st.text_input("Enter the topic you want to write about:")
    
    if topic_query:
        # Initialize session state for storing data
        if 'current_topic' not in st.session_state:
            st.session_state.current_topic = None
        if 'df_serp' not in st.session_state:
            st.session_state.df_serp = None
        if 'df_results' not in st.session_state:
            st.session_state.df_results = None
        if 'targeting_keywords' not in st.session_state:
            st.session_state.targeting_keywords = None
            
        # Only fetch new data if topic has changed
        if topic_query != st.session_state.current_topic:

            # Step 2: SERP Data
            with st.spinner("Fetching live SERP Data..."):
                time.sleep(1.5)
                st.session_state.df_serp = get_serpapi_data(topic_query, st.session_state.SERPAPI_KEY)
                if not st.session_state.df_serp.empty:
                    with st.expander(f"SERP Results of: ''{topic_query}''"):
                        st.dataframe(st.session_state.df_serp)
            
        # Step 3: SEMrush Data && Step 4: Content Retrieval
            st.session_state.df_results = st.session_state.df_serp.copy()
            
            with st.spinner("Fetching SEMRush Data..."):
                time.sleep(1.5)
                st.session_state.targeting_keywords = process_semrush_data(
                    st.session_state.df_results, 
                    st.session_state.SEMRUSH_API_KEY
                )
            
            contents = []
            for idx, link in enumerate(st.session_state.df_results['Link']):
                with st.spinner(f"Fetching content from link {idx + 1}/{len(st.session_state.df_results['Link'])}: {link}"):
                    contents.append(fetch_content(link, st.session_state.JINA_API_KEY))
            st.session_state.df_results['Content'] = contents
            
            # Update current topic
            st.session_state.current_topic = topic_query

        
        # Display stored data
        with st.expander(f"🔍 Top Ranking Pages, Keyword Data & Page Content"):
            df_dict = st.session_state.df_results.to_dict('records')
            for item in df_dict:
                item['Content'] = [item['Content']]
            st.json(df_dict, expanded=False)
            # st.dataframe(st.session_state.df_results)
        
        with st.expander(f"📊 Data Analysis of Top Ranking Sites"):
            st.dataframe(st.session_state.targeting_keywords)


        
        # Steps 5-10: Content Generation
        if st.button("Generate Content"):
            with st.spinner("Analyzing content..."):
                content_analysis = generate_content_analysis(topic_query, st.session_state.df_results, client)
                with st.expander("🧐 Content Analysis"):
                    st.markdown(content_analysis)
            
            with st.spinner("Creating content plan..."):
                content_plan = generate_content_plan(topic_query, st.session_state.targeting_keywords, content_analysis, client)
                with st.expander("🗂️ Content Plan"):
                    st.markdown(content_plan)
            
            with st.spinner("Writing draft..."):
                content_draft = generate_content_draft(content_plan, content_analysis, client)
                with st.expander("✍️ Content Draft"):
                    st.markdown(content_draft)
            
            with st.spinner("Editorial review..."):
                proofread_draft = proofread_content(content_draft, content_plan, content_analysis, client)
                with st.expander("📝 Refined Article"):
                    st.markdown(proofread_draft)
            
            with st.spinner("Generating SEO recommendations..."):
                seo_recommendations = generate_seo_recommendations(proofread_draft, st.session_state.targeting_keywords, client)
                with st.expander("🤖 SEO Recommendations"):
                    st.markdown(seo_recommendations)
            
            with st.spinner("Preparing final deliverable..."):
                final_deliverable = generate_final_deliverable(
                    proofread_draft, seo_recommendations, st.session_state.targeting_keywords, 
                    st.session_state.df_serp, content_analysis, client
                )
                
                st.subheader("🎯 Final Deliverable")
                st.caption("Try copy/paste into https://markdownlivepreview.com/ if the output is not formatted properly.")
                st.divider()
                st.markdown(final_deliverable)

if __name__ == "__main__":
    main()