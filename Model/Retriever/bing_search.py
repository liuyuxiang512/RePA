import requests
from diskcache import Cache
import logging

logging.basicConfig(level=logging.INFO)

'''
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
'''

# subscription_key = os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY']


def bing_search(query: str, max_ret_topk: int, subscription_key: str, exclude_domains: list[str]):
    q_key = "Topic Search: " + query + ' ' + ' '.join([f'-site:{d}' for d in exclude_domains])
    cache = Cache('.bing_search_cache')

    if q_key in cache:
        logging.info(f'Bing CACHE: {q_key}')
        return cache[q_key]
    else:
        endpoint = "https://api.bing.microsoft.com/v7.0/search"

        # Modify query based on domain
        query = query + ' ' + ' '.join([f'-site:{d}' for d in exclude_domains])
        params = {
            'q': query,
            'mkt': 'en-US',
            'responseFilter': ['Webpages'],
            'count': max_ret_topk,
            'safeSearch': 'Off',
            'setLang': 'en-US'
        }
        # Construct a request
        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
            'Pragma': 'no-cache',
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; Touch; rv:11.0) like Gecko',
            'X-MSEdge-ClientID': '128.2.211.82',
        }
        # Call the API
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            response = response.json()
            cache[q_key] = response
            return response
        except Exception as ex:
            logging.error(f'EXCEPTION: {ex}')
            return None


def parse_response(response, exclude_domains):
    results = []
    if response and 'webPages' in response and 'value' in response['webPages']:
        for page in response['webPages']['value']:
            result = {
                'url': page['url'],
                'name': page['name'],
                'snippet': page['snippet']
            }
            exclude = True
            for d in exclude_domains:
                if d in page['url']:
                    exclude = False
                    break
            if exclude:
                results.append(result)
    return results


def search_bing_batch(queries: list[str], kwargs):
    batch_results = []
    for query in queries:
        response = bing_search(
            query=query,
            max_ret_topk=kwargs["topk_Engine"],
            subscription_key=kwargs["BING_SEARCH_V7_SUBSCRIPTION_KEY"],
            exclude_domains=kwargs["exclude_domains"]
        )
        results = parse_response(response=response, exclude_domains=kwargs["exclude_domains"])
        batch_results.append(results)

    return batch_results
