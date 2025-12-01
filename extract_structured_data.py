import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class CheckboxGroup:
    question: str
    value: Optional[bool]
    page_number: int


@dataclass
class TextContent:
    text: str
    html: Optional[str] = None
    page_number: int = 0


@dataclass
class ImageContent:
    page_number: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class HeaderContent:
    text: str
    page_number: int


@dataclass
class Section:
    title: str
    page_number: int
    content: List[Union[str, TextContent, CheckboxGroup, ImageContent, HeaderContent]] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)


def detect_checkbox_state(element: Dict[str, Any], prev_element: Optional[Dict[str, Any]] = None, next_element: Optional[Dict[str, Any]] = None) -> Optional[str]:
    html = element.get('metadata', {}).get('text_as_html', '')

    if '<input class="Checkbox"' in html:
        is_checked = 'checked' in html

        # Look at previous element first (checkboxes usually come after Yes/No labels)
        if prev_element:
            prev_text = prev_element.get('text', '').strip()
            if prev_text == 'Yes' and is_checked:
                return 'checked_yes'
            elif prev_text == 'No' and is_checked:
                return 'checked_no'
            elif prev_text in ['Yes', 'No'] and not is_checked:
                return 'unchecked'

        # Also check next element to determine if it's Yes or No
        if next_element:
            next_text = next_element.get('text', '').strip()
            if next_text == 'Yes' and is_checked:
                return 'checked_yes'
            elif next_text == 'No' and is_checked:
                return 'checked_no'
            elif next_text in ['Yes', 'No'] and not is_checked:
                return 'unchecked'

    return None


def extract_structured_data(json_path: str) -> List[Section]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    sections = []
    current_section = None
    pending_checkbox_question = None
    i = 0

    # Create a default section if no Title elements exist
    has_title = any(elem.get('type') == 'Title' for elem in data)
    if not has_title and data:
        current_section = Section(
            title="Document Content",
            page_number=data[0].get('metadata', {}).get('page_number', 0) if data else 0
        )

    while i < len(data):
        element = data[i]
        prev_element = data[i - 1] if i > 0 else None
        next_element = data[i + 1] if i + 1 < len(data) else None

        element_type = element['type']
        text = (element.get('text') or '').strip()
        page_number = element['metadata'].get('page_number', 0)

        # New section (Title element)
        if element_type == 'Title':
            if current_section and current_section.content:
                sections.append(current_section)

            current_section = Section(
                title=text,
                page_number=page_number
            )

            # Also add Title as a text item in the section
            if text:
                html = element.get('metadata', {}).get('text_as_html', '')
                text_content = TextContent(text=text, html=html, page_number=page_number)
                current_section.content.append(text_content)

            i += 1
            continue

        # Check for standalone checkbox elements
        checkbox_state = detect_checkbox_state(element, prev_element, next_element)

        if checkbox_state:
            # Look back to find the question
            if pending_checkbox_question:
                question_text = pending_checkbox_question
                pending_checkbox_question = None
            else:
                # Look backwards for a question
                question_text = "Unknown question"
                for j in range(i - 1, max(0, i - 5), -1):
                    prev_text = data[j].get('text', '').strip()
                    if '?' in prev_text or prev_text.endswith(':'):
                        question_text = prev_text
                        break

            # Determine checkbox value
            checkbox_value = {
                'checked_yes': True,
                'checked_no': False
            }.get(checkbox_state, None)

            checkbox = CheckboxGroup(
                question=question_text,
                value=checkbox_value,
                page_number=page_number
            )

            if current_section:
                current_section.content.append(checkbox)

            if next_element and next_element.get('text', '').strip() in ['Yes', 'No']:
                i += 2
                continue
            i += 1
            continue

        if element_type == 'Image':
            if current_section:
                image = ImageContent(
                    page_number=page_number,
                    metadata=element.get('metadata', {})
                )
                current_section.content.append(image)
            i += 1
            continue

        if element_type == 'Header':
            if current_section and text:
                header = HeaderContent(
                    text=text,
                    page_number=page_number
                )
                current_section.content.append(header)
            i += 1
            continue

        if text and element_type not in ['Image', 'Header']:
            if text not in ['Yes', 'No']:
                if current_section:
                    html = element.get('metadata', {}).get('text_as_html', '')
                    text_content = TextContent(text=text, html=html, page_number=page_number)
                    current_section.content.append(text_content)
                    if '?' in text or text.endswith(':'):
                        pending_checkbox_question = text

        i += 1

    if current_section and current_section.content:
        sections.append(current_section)

    return sections


def export_to_flat_list(sections: List[Section]) -> List[Dict[str, Any]]:
    
    flat_list = []

    def process_sections(sections: List[Section]):
        for section in sections:
            section_title = section.title

            for item in section.content:
                if isinstance(item, CheckboxGroup):
                    # Checkbox item
                    flat_list.append({
                        "type": "checkbox",
                        "question": item.question,
                        "value": item.value,
                        "section": section_title,
                        "page_number": item.page_number
                    })
                elif isinstance(item, ImageContent):
                    # Image item
                    flat_list.append({
                        "type": "image",
                        "section": section_title,
                        "page_number": item.page_number
                    })
                elif isinstance(item, HeaderContent):
                    # Header item
                    flat_list.append({
                        "type": "text",
                        "value": item.text,
                        "section": section_title,
                        "page_number": item.page_number
                    })
                elif isinstance(item, TextContent):
                    # Text item
                    flat_list.append({
                        "type": "text",
                        "value": item.text,
                        "section": section_title,
                        "page_number": item.page_number
                    })

            # Process subsections recursively
            if section.subsections:
                process_sections(section.subsections)

    process_sections(sections)
    return flat_list


def process_directory(input_dir: str = "converted", output_dir: str = "extracted"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return

    # Find all JSON files recursively
    json_files = list(input_path.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found in '{input_dir}'")
        return

    print(f"Found {len(json_files)} JSON file(s) to process\n")

    for json_file in json_files:
        try:
            relative_path = json_file.relative_to(input_path)

            output_file = output_path / relative_path
            if output_file.name.endswith('.pdf.json'):
                output_file = output_file.parent / output_file.name.replace('.pdf.json', '.json')

            # Create output directory if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)

            print(f"Processing: {relative_path}")

            # Extract data
            sections = extract_structured_data(str(json_file))
            flat_list = export_to_flat_list(sections)

            # Save to output file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(flat_list, f, indent=2, ensure_ascii=False)

            # Count items by type
            type_counts = {}
            for item in flat_list:
                item_type = item['type']
                type_counts[item_type] = type_counts.get(item_type, 0) + 1

            # Format counts for display
            counts_str = ', '.join([f"{count} {type_name}{'s' if count != 1 else ''}"
                                   for type_name, count in sorted(type_counts.items())])

            print(f"{len(flat_list)} items ({counts_str})")
            print(f"{output_file}\n")

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}\n")
            continue

if __name__ == "__main__":
    # Process all JSON files in converted directory
    process_directory(input_dir="converted", output_dir="extracted")
