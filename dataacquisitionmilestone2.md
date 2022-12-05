```python
from google_images_search import GoogleImagesSearch
import os
import shutil
```


```python
# !pip install Google-Images-Search
```


```python
list_of_items = ['Room heaters','Bed','Television', 'Cabinet', 'Treadmill', 
                 'Vase', 'Generator', 'Sofa', 'Carpets', 'Floor']
```


```python
api_keys = ['AIzaSyBhvJ_boV-yCudo8AAqZuZrzBVrOKV-r3M',
            'AIzaSyBsZJWBhoqcqzjZoCkHTQKYFcoNpVtuUNU',
            'AIzaSyASipLtaBxYolXmoo63qlTY3lLZgQByRE0',           
            'AIzaSyCuuot3hsf4EsIbGt4UwEHU64V8GgskV-E',        
            'AIzaSyBvNKrJRB494JH2y-OWwMwjA7oZ4ppkXaQ',   
            'AIzaSyDAawMs7tcbPyOMyRtl3B_zPY9hm2CpZqg',   
            'AIzaSyDrX20ACxsqI5qp4gRzHVy8WiJ5wzRj9bM',     
            'AIzaSyBpSy_3whHaKixQDX4UPwosUJ03KWVkf7g',  
            'AIzaSyCrCHhveZFNE9p2aPErDRikLD90Ug0D0Kk',       
            'AIzaSyA7q_jeE0I3ynRuFP4vFwgIMht3qhhiBe4'          
          ]
```


```python
api_count = 0
```


```python
def create_dir(file_name,item=True):
    suffix = 0
    if item:
        try:
            os.mkdir(f'images/{file_name}')
        except:
            os.remove(f'{file_name}')
            create_dir(file_name,item=True)
 
    else:
        try:
            os.mkdir(f'{file_name}')
        except:
            shutil.rmtree(f'{file_name}')
            create_dir(file_name,item=False)
```


```python
def fetch_item_images_helper(item_name, api_count):
    _search_params = {
        'q': f'{item_name}',
        'num': 100,'fileType': 'jpg|png',
    }
    gis = GoogleImagesSearch(api_keys[api_count], 'b2dad1940a4a74e80')
    gis.search(search_params=_search_params, path_to_dir=f'images/{item_name}')

```


```python
def fetch_item_images(item_name):
    global api_count
    create_dir(item_name)
    try:
        fetch_item_images_helper(item_name,api_count)
        print(f"Completed for {item_name}")
    except:
        try:
            if api_count< len(api_keys)-1:
                api_count = api_count + 1
                print(f"Changed API to {api_count}")
                fetch_item_images_helper(item_name,api_count)
            else:
                return 0
        except:
            return 0
```


```python
create_dir("images", item = False)
for each_item in list_of_items:
    fetch_item_images(each_item)
```

    Completed for Room heaters
    Completed for Bed
    Completed for Television
    Completed for Cabinet
    Completed for Treadmill
    Completed for Vase
    Completed for Generator
    Completed for Sofa
    Completed for Carpets
    Completed for Floor



```python

```
