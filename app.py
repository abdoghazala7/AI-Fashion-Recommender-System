import streamlit as st
from PIL import Image
import requests
from io import BytesIO

from Get_LLM_response import LLMRecommender
from Get_Ai_Image_Description import get_image_description

st.set_page_config(page_title="Fashion Recommender", layout="wide")

# -------- Helper Functions --------
def get_image_and_description_by_index(image_index: int) -> tuple:
    dataset = "tomytjandra/h-and-m-fashion-caption"
    config = "default"
    split = "train"
    
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={dataset}"
        f"&config={config}"
        f"&split={split}"
        f"&offset={image_index}"
        f"&limit=1"
    )

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    
    data = response.json()
    
    try:
        row = data["rows"][0]["row"]
        image_url = row["image"]["src"]
        description = row["text"]
        
        image_response = requests.get(image_url)
        image = Image.open(BytesIO(image_response.content))
        
        return image, description
    except (IndexError, KeyError) as e:
        raise ValueError("Invalid response structure or index out of range") from e

def load_image_from_url(url: str):
    try:
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    except Exception:
        return None

def generate_enhanced_query(user_query: str, image_description: str = None):
    final_query = f"{user_query}\nItem Description: {image_description}" if image_description else user_query
    llm_recommender = LLMRecommender()
    return llm_recommender.get_recommendations(final_query)

def display_recommendations(recommendations):
    st.subheader("🎯 Recommended Items")
    st.info("Finding items using the enhanced query.")
    
    with st.spinner("Generating Recommended Items..."):
        results = recommendations.get("results", [])
        if not results:
            st.error("Unfortunately 🙁, no recommendations were found in our data.")
            return

        cols = st.columns(4)
        for idx, rec in enumerate(results[:4]):
            try:
                img, desc = get_image_and_description_by_index(int(rec["id"]))
                with cols[idx % 4]:
                    st.image(img, use_container_width=True)
                    st.markdown(f"**Description:** {desc}")
            except Exception as e:
                st.warning(f"Failed to load item {rec['id']}: {e}")

def process_query(user_query: str, image_description: str = None):
    if not user_query:
        st.info("Please enter a description to get recommendations.")
        return

    if st.button("Get Enhanced Query"):
        try:
            enhanced_query, recommendations = generate_enhanced_query(user_query, image_description)
            st.session_state.enhanced_query = enhanced_query
            st.session_state.recommendations = recommendations
        except Exception as e:
            st.error(f"❌ Failed to get recommendations: {e}")
            return

    if "enhanced_query" in st.session_state:
        st.subheader("📝 Enhanced Query")
        st.success(st.session_state.enhanced_query)

        if st.button("Get Recommendations"):
            display_recommendations(st.session_state.recommendations)

# -------- Streamlit App UI --------
st.title("🤵 AI Fashion Recommender")
st.info("""
**📝 How to Use This App:**

1. Upload an image, paste an image URL, or type a fashion-related query.
2. **After typing your query, press Enter.**
3. Click **'Get Enhanced Query'** to let AI improve your search.
4. After the enhanced query appears, click **'Get Recommendations'** to see suggested items.
""")


input_method = st.radio("Select input method:", ["Upload Image", "Image URL", "Text Query"])

image = None
image_description = None
user_query = ""

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        image_description = get_image_description(uploaded_file, is_local=True)
        if image_description:
            st.markdown("### 📝 Analysis Results")
            st.markdown(image_description)
            user_query = st.text_input("Describe what you're looking for with this item:")
            process_query(user_query, image_description)
        else:
            st.error("❌ Error: Unable to analyze the image. Please try again.")

elif input_method == "Image URL":
    url = st.text_input("🔗 Enter image URL:")
    if url:
        image = load_image_from_url(url)
        if image:
            st.image(image, caption="Image from URL", width=300)
        else:
            st.error("❌ Error: Unable to load image.")
        
        image_description = get_image_description(url, is_local=False)
        if image_description:
            st.markdown("### 📝 Analysis Results")
            st.markdown(image_description)
            user_query = st.text_input("Describe what you're looking for with this item:")
            process_query(user_query, image_description)
        else:
            st.error("❌ Error: Unable to analyze the image. Please try again.")

elif input_method == "Text Query":
    user_query = st.text_input("Enter your fashion request:")
    process_query(user_query)
