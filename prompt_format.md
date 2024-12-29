# Documentation: `format_text_input` Method

The `format_text_input` method is part of the `BgeGemma2MultimodalProcessor` class and is responsible for generating formatted text and vision (image) input combinations, allowing support for text-only, image-only, and multimodal embeddings. 

This documentation outlines the key functionality, parameter descriptions, and examples of the method's behavior.

---

## Method Signature

```python
def format_text_input(
    self, 
    text: TextInput = None, 
    image=None, 
    instruct: TextInput = None
) -> TextInput
```

---

## Parameters

### `text` (Optional)
- **Type:** `TextInput`  
- **Description:**  
  The primary text input. Can be a string or tokenized text such as prompts, sentences, or instructions.
- **Example:** `"Find images of cats"`

### `image` (Optional)
- **Type:** N/A  
- **Description:**  
  Represents visual inputs (images). Its inclusion generates multimodal embeddings.
- **Example:** An image or image data.

### `instruct` (Optional)
- **Type:** `TextInput`  
- **Description:**  
  Additional instructions to guide the text/image embedding process.
- **Example:** `"Given a web search query, retrieve relevant images"`

---

## Returns

- **Type:** `TextInput`  
- **Description:**  
  A formatted string output that combines the text and image inputs, enriched with additional tokens (`<vision>`, `<instruct>`, `<query>`) as appropriate for downstream tasks.

---

## Predefined Tokens

To format inputs, the method utilizes the following special tokens:

| Token              | Representation | Description                             |
|--------------------|----------------|-----------------------------------------|
| `<vision>`         | `self.IMAGE_TOKEN` | Marks the presence of image-based input.|
| `<instruct>`       | `self.TEXT_INSTRUCT_TOKEN` | Precedes instructions within the input. |
| `<query>`          | `self.QUERY_TOKEN` | Marks the inclusion of search queries.  |

---

## Input Formatting Combinations

### 1. **Image-Only Embedding**
- **Scenario:** No `text` provided; method relies on `image` and `instruct`.  
- **Behavior:**
  - If `instruct` is present:
    - Format: `"<vision><instruct>{instruction_text}"`
  - If no `instruct` is provided:
    - Format: `"<vision>"`  
- **Example:**
  ```plaintext
  Input: 
    text=None, image="image_data", instruct="Retrieve similar images"
  Output: 
    <vision><instruct>Retrieve similar images
  ```

---

### 2. **Text-Only Embedding**
- **Scenario:** No `image` provided; method formats only `text` with optional `instruct`.  
- **Behavior:**
  - If `instruct` is present:
    - Format: `"<instruct>{instruction_text}<query>{text}"`
  - If no `instruct` is provided:
    - Format: `{cleaned_text}`  
- **Example:**
  ```plaintext
  Input: 
    text="My last bill", image=None, instruct="Retrieve billing info"
  Output: 
    <instruct>Retrieve billing info<query>My last bill
  ```

---

### 3. **Image and Text Embedding**
- **Scenario:** Both `image` and `text` provided with optional `instruct`.  
- **Behavior:**
  - If `instruct` is present:
    - Format: `"<vision><instruct>{instruction_text}<query>{text}"`
  - If no `instruct` is provided:
    - Format: `"<vision>{text}"`  
- **Example:**
  ```plaintext
  Input: 
    text="Find similar items", image="image_data", instruct="Help identify similar objects"
  Output: 
    <vision><instruct>Help identify similar objects<query>Find similar items
  ```

---

## Helper Function: Token Cleaning

The method uses a helper function `remove_tokens` to clean conflicting special tokens (`<vision>`, `<instruct>`, and `<query>`) from input text and instructions. This ensures consistent formatting across all scenarios.

---

## Example Usage

```python
# Example 1: Text only
processor.format_text_input(
    text="Retrieve the last bill",
    image=None,
    instruct="Search query for financial records"
)
# Output: 
# <instruct>Search query for financial records<query>Retrieve the last bill

# Example 2: Image only
processor.format_text_input(
    text=None,
    image="image_data",
    instruct="Classify this image"
)
# Output: 
# <vision><instruct>Classify this image

# Example 3: Multimodal (Image and Text)
processor.format_text_input(
    text="Find similar styles", 
    image="image_data", 
    instruct="Identify fashion trends"
)
# Output: 
# <vision><instruct>Identify fashion trends<query>Find similar styles
```

---

## Use Cases

### 1. **Query-Based Vision/Language Tasks**
Applications leveraging both text queries and visual inputs, such as:
- Image classification using both vision and text context.
- Search engines for retrieving query-specific images.

### 2. **Instruction-Driven Multimodal Systems**
Use cases where instructions refine processing:
- Explaining detailed actions like "Compare two images."
- Annotating both text and image embeddings with instructions.

### 3. **Flexible Multimodal Embeddings**
Supports tasks that require working with either:
- **Text-only inputs**
- **Image-only inputs**
- **Combined text and image inputs**

---

## Conclusion

The `format_text_input` method is essential for the `BgeGemma2MultimodalProcessor` class. It ensures consistent preprocessing for downstream multimodal tasks by producing unified embeddings. The inclusion of special tokens and support for diverse input formats makes this method highly flexible for vision, language, or multimodal AI tasks.