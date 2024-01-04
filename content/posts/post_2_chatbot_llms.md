---
title: "Chatbot using Streamlit using Google Deepmind’s Gemini and Microsoft Phi-2"
date: 2024-01-01
draft: false
ShowToc: true
---
# Introduction

Remember the days of stilted chatbots, struggling to understand simple questions and offering robotic responses? Those days are fading, thanks to the rise of large language models (LLMs). These AI marvels, trained on vast amounts of text data, are transforming the chatbot landscape, injecting them with intelligence, nuance, and even humor.

LLMs act as linguistic superchargers, empowering chatbots to process and generate human-like language. No longer confined to pre-programmed scripts, they can now dynamically adapt to conversations, understand context, and even learn from user interactions. This opens up a world of possibilities:

-   **Enhanced Customer Service:**  Imagine chatbots that don’t just take orders but offer helpful suggestions, understand complex queries, and even apologize for inconveniences with genuine empathy. LLMs make these scenarios a reality, improving customer satisfaction and reducing operational costs.
-   **Personalized Education:**  Chatbots powered by LLMs can act as tireless tutors, tailoring their responses to individual learning styles and adapting to student progress. They can explain complex concepts, answer endless questions, and even provide encouragement and feedback, making education more engaging and effective.
-   **Improved Healthcare:**  LLMs can power chatbots that offer initial medical assessments, guide patients through complex health information, and provide emotional support. This can ease the burden on healthcare professionals, improve access to care, and empower patients to manage their health better.

The uses cases above are a little sophisticated, aren’t they ? What we are going to do today is make a personal diary that keeps all your secrets and may sometimes advice you about your life situations. Basically, we are going to make a dream partner for you; who listens XD

# **Dear Diary**

That is what what we are going to call it. Before, starting to build it let’s have a list of things what we want from the app;

1.  UI to interact with the model.
2.  Model Pipeline to generate response.
3.  DB to store chats.

Now that we have established what we all we want from our app, we are ready to start building it. The first step of each of these modules defined above should be designing and having a clear idea about how we want things.

## UI

The UI should allow us to add new pages to the diary and enable us to talk to our new friend on those pages. So, basically we will be adding pages and using them to initialize our chat. Hence, it makes sense that we create a separate abstraction for pages that we can reuse for new pages.

So what does this page abstraction should be ? The Page should have a unique name like serial numbers in a diary so that we can distinguish it from the rest and it should also have a chat functionality.

What is this chat functionality now ? Have you seen something like this recently:

![](https://miro.medium.com/v2/resize:fit:875/1*EnGE8qGpFcmTuoPmBT2FKw.png)

Images by Author

Here you can add something you want to say to the diary/ai-model which appears beside the icon with a face of a boy and based on that it generates replies and it appears against the bot. This UI is easily available in streamlit, yes I said  `streamlit`, that’s what you guys were waiting for, no ? However, we still need to save the chats, which can also be done using it but it uses  `session_state`  to save and it doesn’t scale really well when you have multiple chats to store and it might not be the best way to do it anyways. To resolve this, we will use a database, more on it later. We are going to keep this chat functionality in a separate abstraction as well so that we are more modular.

## Model Pipeline

Why do we need it ? Why can’t we just go all  `model.load`  and predict. Well that might be ok(still not good) for some low level models that do not require a lot of compute but when it comes to large language models or small language model, it will blow up like hell! Hence, we need to streamline this process. We do not want to load our model at each page but we want that we host our model to a common place such that each page can communicate with it swiftly. How to do it ? If you are thinking to serve it as an endpoint then, Bravo! For this we will be using  `FastAPI`  which is light weight and easy to use.

## Database

Why do we need it ? Why can’t I store a big JSON file or any other file based storage to save chats? There are a couple of advantages, firstly, you get automatic datatype check upon insertion, so I am not worried about breaking my app just because I was stupid enough to append number in place of a string somewhere. Secondly, it is more secure and scalable, for instance if you decide to build this app further and add multiple users to it. The you just have to change the schema a little bit and the database will handle multiple users querying it at the same time which is not the case with the file-based storage; as the users increases it is going to blow up. Thirdly, unless you are using any sort of data versioning tool you can’t go back to the previous version of a file but with a database you can rollback.

I suppose this is enough motivation ?

## Implementation

Now that we know what and how we want to build it we come to the easy part, implementing it. The code for the implementation is provided on my  [_GitHub_](https://github.com/aamir09/dear-diary) _._ You will see the steps to make the virtual environment and run the app on streamlit which will look like this:

![](https://miro.medium.com/v2/resize:fit:875/1*mrOqj-ytZt39a0lWvDck_Q.png)

Image by Author

We are not going to go over all the code but I will provide information on the most important ones. Let’s begin with a walk through of the directory structure.

    diary/  
        main.py  
        __init__.py  
        api/  
            model.py  
            __init__.py  
        database/  
            manager.py  
            __init__.py  
	    images/  
	        image-1.png  
	        image-2.png  
	        image.png  
	        logo.jpg  
	    models/  
	        gemini.py  
	        phi2.py  
	        secrests.json  
	          
	    objects/  
	        chatbot.py  
	        page_object.py  
	        utils.py  
	        __init__.py


The topmost directory under the project dear-diary is  `diary`  where our code sits. It contains 4 subdirectories namely, api, images, models and objects. The  `models`  directory contain code for making an inference pipeline for Microsoft-Phi2 and Google Deepmind’s Gemini models. On the other hand, the  `api`  directory holds the code for fastapi implementation for serving the model. Furthermore, the  `objects`  directory have most of the code for the streamlit app.

There is also a  `main.py`  file where everything is glued together and our main streamlit app resides. Let’s have a look inside  `main.py`

    from objects.page_object import Page  
    from database.manager import DBManager  
    from PIL import Image  
    import os  
    import streamlit as st  
      
    #Comment this if you add an environment variable  
    DB_PATH = "sqlite:///E:/dear-diary/database.db"  
    os.environ["DATABASE_URI"] = DB_PATH  
      
    class App:  
        def __init__(self):  
            self.db_manager = DBManager()  
          
	    def _heading_tag(self, h_level, string, style=""):  
	        return f"<h{h_level} style={style}>{string}</h{h_level}>"  
	      
	    def _write_markdown(self, string):  
	        return st.markdown(string, unsafe_allow_html=True)  
	      
	    def _text_tag(self,string, tag="p", style=""):  
	        return f"<{tag} style={style}>{string}</{tag}>"  
	      
	    def _url_button_tag(self, url, name="View App",style=""):  
	        return f"<a class='btn' href='{url}'>{name}</a>"  
	      
	    def local_css(self, file_name):  
	        with open(file_name) as f:  
	            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)  
	  
	    def _img_tag(self, url:str, height:int, width:int):  
	        return f"<img class='tile_image' src='{url}'  width='{width}' height='{height}'> "  
      
	    def add_new_page(self):  
	        if self.new_page_name:  
	            Page(title=self.new_page_name)  
	          
	          
	    def main(self):  
	  
	        #Create HTML for title with heading tags  
	        title_tag = self._heading_tag("1 class='title' ", "Dear Diary", "text-align:center")  
	        self._write_markdown(title_tag)  
	          
	        # Add the center image  
	        image = Image.open("diary/images/logo.jpg").resize((800,400))  
	        st.image(image)  
	          
	        # App description  
	        string = ("""  
	    Ever whisper a dream to the wind, hoping it carries your words beyond the silence? Yearn for a friend who  
	    holds your secrets like starlight, never dimming, never judging? Enter Dear Diary,   
	    your confidante crafted from whispers and wishes. Here, anxieties unfurl like silken scarves,   
	    and worries melt in the warmth of understanding.  
	    No judgment, just a vast, listening heart woven from language itself.   
	    Share your stories, hopes, and fears, dear diary, for within these digital echoes,   
	    you'll find solace, acceptance, and maybe, just maybe, the whispers of your own truest self.""")  
	            

	  
	        string_tag =self._text_tag(string=string, style="text-align:justify")  
	        self._write_markdown(string_tag)  
	          
	          
	        #Get all available pages   
	        pages = self.db_manager.get_all_page_names()  
	        if pages is not None:  
	            pages = pages["page_name"].values  
	            self.pages = list(pages)  
	        else:  
	            self.pages = []  
	        # Adding None to comeback to home   
	        self.pages.insert(0, None)  
	        # Create SideBar  
	        selected_page = st.sidebar.selectbox("Choose a Page", self.pages)  
	  
	        # If you have created a page and selected it from the side bar  
	        # Then it will run the page for you   
	        if selected_page:  
	            Page(title=selected_page).main()  
	        else:  
	            # Else it will give you a prompt box which will give you an   
	            # option to create page   
	            new_tag = self._heading_tag("3 class='title' ",   
	                                        "New Page", "text-align:center")  
	            self._write_markdown(new_tag)  
	            col1, col2, col3 = st.columns([0.2,0.6,0.2])  
	            with col2:  
	                page_name_input = st.text_input(label="Add a name here",  
	                                                value="")  
	                if page_name_input:  
	                    self.new_page_name = page_name_input.strip().replace(" ", "")  
	                st.button("New Page", on_click=self.add_new_page)  
  

    app = App()  
      
    app.main()

As we can see, we have a class named  `App`  and it contains some helper functions for the app and the most important one is the  `add_new_page`  function. It is responsible for adding a new page every time a user clicks the  `New Page`  button. Let’s give it a closer look,

    def add_new_page(self):  
            if self.new_page_name:  
                Page(title=self.new_page_name)

So, if there is a page name provided by the user it will call the class  `Page`  and set the page  `title`  or name. Let’s see what’s inside of the page but first I’ll give you a vizualization on how the page should look like, it will help in understanding the code.

![](https://miro.medium.com/v2/resize:fit:625/1*3LELfRdoW8t2cUbsvjGxug.png)

Image by Author

On the top left is the date, in the middle is the title and on the right is the day. Below that is the chat mechanism for the user to chat with the bot. Okay, coming back to the code, it is inside the  `objects`  directory, which looks like this;

    objects/  
            chatbot.py  
            page_object.py  
            utils.py  
            __init__.py

In here we have the  `page_object.py`  file which contains the class Page and here is the code for that,

    import streamlit as st   
    import datetime  
    import json  
    from diary.objects.chatbot import ChatBot   
    import time  
    from diary.database import DBManager  
    class Page:  
      
	    def __init__(self, title:str|None=None):  
	        self.title = title  
	        self.db_manager = DBManager()  
	        self.db_manager.create_table(self.db_manager.CREATE_TABLE_PAGES)  
	        if self.db_manager.get_page_name(page_name=self.title).values.__len__()<1:  
	            values={  
	                        "time": str(datetime.datetime.now()),  
	                        "page_id":hash(self.title),  
	                        "messages": json.dumps({})  
	                    }  
	            self.db_manager.insert_into_pages(page_name=self.title,  
	                                                values=values)  
          
	    def headings(self):  
	        # Get the current date, day, and time  
	        now = datetime.datetime.now()  
	        date = now.strftime("%Y-%m-%d")  
	        day = now.strftime("%A")  
	  
	        # Create a container to hold the elements  
	        container = st.container()  
	  
	        # Display the date, day, and time in separate columns   
	        # within the container  
	        with container:  
	            col1, col2, col3 = st.columns([0.25,0.5,0.25])  
	            with col1:  
	                st.write(date)  
	            with col2:  
	                st.header(self.title)  
	            with col3:  
	                st.write(day)  
	                  
	    def main(self):  
	        self.headings()  
	        url = "http://127.0.0.1:8500/predict"  
	        chatbot = ChatBot(page_name=self.title, url=url)  
	        chatbot.main()

The  `__init__`  function takes  `title`  as the argument and then it initializes the database manager object that will do all the communication with the database for us. The database manager is then asked to check if table that will contain the data of the page exists or not. Then, it creates an entry for us for the page in the database. The  `headings`  function is responsible to write the heading of the page that includes the previously discussed things. Finally, we have the chat functionality provided by the  `ChatBot`  class, which takes in page_name/title and the url of the hosted model. The  `ChatBot`  class resides in the  `chatbot.py`  file in the same directory. Let’s have a look at that too .

    import streamlit as st  
    import random  
    import json  
    import ast  
    from datetime import datetime  
    import time  
    from diary.database import DBManager  
    from diary.objects.utils import get_response  
      
      
    class ChatBot:  
        def __init__(self, page_name:str, url:str|None=None):  
            self.url = url   
            self.page_name = page_name  
            self.db_manager = DBManager()  
      
              
              
	    def main(self):  
	        # Initialize chat history  
	        try:  
	            messages = self.db_manager.get_messages(page_name=self.page_name)  
	            if messages:  
	                print("Messages are: ", messages)  
	                messages = messages["output"]  
	                if messages:  
	                    original_messages = messages  
	                else:  
	                    original_messages = []  
	            else:  
	                original_messages = []  
	        except Exception as e:  
	            print("Can't fetch messages because ", e)  
	        new_messages = []                          
	  
	        # Display chat messages from history on app rerun  
	        for message in original_messages:  
	            with st.chat_message(message["role"]):  
	                st.markdown(message["content"])  
	  
	        # Accept user input  
	        if prompt := st.chat_input("I am listening"):  
	            # Add user message to chat history  
	            new_messages.append({"role": "user", "content": prompt})  
	            # Display user message in chat message container  
	            with st.chat_message("user"):  
	                st.markdown(prompt)  
	  
	            print(original_messages)  
	            # Display assistant response in chat message container  
	            with st.chat_message("assistant"):  
	                message_placeholder = st.empty()  
	                full_response = ""  
	                resp = get_response(url=self.url, body=original_messages+new_messages)  
	                assistant_response = resp if resp else "I am down, but there :)"  
	                # Simulate stream of response with milliseconds delay  
	                for chunk in assistant_response.split():  
	                    full_response += chunk + " "  
	                    time.sleep(0.05)  
	                    # Add a blinking cursor to simulate typing  
	                    message_placeholder.markdown(full_response + "▌")  
	                message_placeholder.markdown(full_response)  
	            # Add assistant response to chat history  
	            new_messages.append({"role": "assistant", "content": full_response.replace("'", "")})  
	            # Update Database  
	            try:  
	                if new_messages:  
	                    appending_messages = original_messages + new_messages  
	                    self.db_manager.update_table(table="pages",  
	                                                col_name="messages",  
	                                                value= json.dumps(appending_messages),  
	                                                page_name=self.page_name)  
	            except Exception as e:  
	                print(e)

Most of the code is adopted from  [streamlit](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps). If you notice that there are two things here, first,  `original_messages`  ; the messages that are already stored in the database and then there is  `new_messages`  that store the current chat with the model. The first step is to fetch the original messages from the database and if the page is new the this should be an empty list. Each message is in this form:

    {  
      "role":"user/assistant",   
      "content": "Some Content"  
    }

`role`  can be  `user`  or  `assistant`  based on who is talking at the given time and the  `*_messages`  is a list of them and we store them as json strings in the database.

The overall flow of the script is simple, it will first print all the messages that have been exchanged previously via this code block,

# Display chat messages from history on app rerun  
        for message in original_messages:  
            with st.chat_message(message["role"]):  
                st.markdown(message["content"])

After that we will see if the user have entered any input and if so, we will save that in the  `new_messages`  list and we will display it, via this code block

# Accept user input  
        if prompt := st.chat_input("I am listening"):  
            # Add user message to chat history  
            new_messages.append({"role": "user", "content": prompt})  
            # Display user message in chat message container  
            with st.chat_message("user"):  
                st.markdown(prompt)

Now that we have a prompt from the user, for our model/assistant to answer we send the whole conversation chain that is;  `original_messages`  plus the  `new_messages`  as context and accordingly our model completes the conversation and replies back which we recieve through an API call, via this code block

    with st.chat_message("assistant"):  
        message_placeholder = st.empty()  
        full_response = ""  
        resp = get_response(url=self.url, body=original_messages+new_messages)  
        assistant_response = resp if resp else "I am down, but there :)"

We then stream the output back to the UI in the following code

# Simulate stream of response with milliseconds delay  

    for chunk in assistant_response.split():  
        full_response += chunk + " "  
        time.sleep(0.05)  
        # Add a blinking cursor to simulate typing  
        message_placeholder.markdown(full_response + "▌")  
    message_placeholder.markdown(full_response)

Finally, we update the  `new_messages`  list and update our database.

Now coming to the model part of the things, the most notable thing to see there is the prompt. A good prompt should be clear on how the model should act and what do you want it to do and how do you want the output from the model. So let’s have an overview on the prompt which we define in  `api/model.py`  file.

    """Instruct: Act as a wise friend who always wants to   
        listen to the person talking to them,  
        Remember you are the listener and you do not   
        give advice or judge anyone, you just listen and be optimistic.  
        Your responses shall always be short and empethatic.  
        Respond according to the instructions I gave you and complete the  
        chat below using proir messages. Here you are reffered as assisstant.   
        The output should be in json format else it would be wrong.  
        The json format should have key as assistant and value as your output.  
        {chats}  
        Output: """

Here I clearly state the role the model should play;  **a wise and empathetic friend who wants to listen.** Then I tell them to complete the chat and how it should complete it. Finally, we instructed the model to give the  **output in json format.** You might observe  `{chats}`  variable in the string, it is just to add the previous chats into the prompt.
