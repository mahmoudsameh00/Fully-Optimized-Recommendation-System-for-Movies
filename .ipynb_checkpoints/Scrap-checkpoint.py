def Search():
    global popularity
    global rete
    global date
    global runtime
    global genres
    global original_language
    x=input('Enter title of the movie PLS : ')
    x=x.replace(' ','+')
    import webbrowser
    url='https://www.themoviedb.org/search?query={}'.format(x)
#     webbrowser.open('https://www.themoviedb.org/search?query={}'.format(x), new=0)
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)

    try:
        button = driver.find_element("xpath","//*[@class='result']")
        button.click()
    except:
        return print('There are no movies that matched your query')

    if driver.current_url.split('/')[3]!='movie':
        return print('it is not a movie')
    else:
        ## get Id of Movie
        i=driver.current_url.split('/')[-1].split('-')[0]
        ## Read API by json
        import requests
        import json
        json_resp=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=9e8098e67bb6c80f099f225ebe3368cc'.format(i))
        data=json.loads(json_resp.text)
        ## Get Rate
        import Rate
        rate=Rate.rate(data['vote_count'],data['vote_average'])
        ## Get Popularity
        popularity=data['popularity']
        ## Get Date
        date=int(data['release_date'].split('-')[0])
        ## Get Original_language
        original_language=data['original_language']
        ## Get Runtime
        runtime=data['runtime']
        ## Get Genres
        genres=[d['name'] for d in data['genres']]
        ## Get overview
        overview=data['overview']
        return print('popularity :',popularity,'\n','rate :',rate,'\n','date :',date,'\n','original_language :',original_language,'\n','runtime :',runtime,'\n','genres :',genres,'\n','overview :',overview),(popularity),(runtime),(date),(rate),(original_language),(genres),(overview)
        
        

        