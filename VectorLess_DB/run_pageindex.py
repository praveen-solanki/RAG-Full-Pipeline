# # import argparse
# # import os
# # import json
# # from pageindex import *
# # from pageindex.page_index_md import md_to_tree
# # from pageindex.utils import ConfigLoader

# # if __name__ == "__main__":
# #     # Set up argument parser
# #     parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
# #     parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
# #     parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

# #     parser.add_argument('--model', type=str, default=None, help='Model to use (overrides config.yaml)')

# #     parser.add_argument('--toc-check-pages', type=int, default=None,
# #                       help='Number of pages to check for table of contents (PDF only)')
# #     parser.add_argument('--max-pages-per-node', type=int, default=None,
# #                       help='Maximum number of pages per node (PDF only)')
# #     parser.add_argument('--max-tokens-per-node', type=int, default=None,
# #                       help='Maximum number of tokens per node (PDF only)')

# #     parser.add_argument('--if-add-node-id', type=str, default=None,
# #                       help='Whether to add node id to the node')
# #     parser.add_argument('--if-add-node-summary', type=str, default=None,
# #                       help='Whether to add summary to the node')
# #     parser.add_argument('--if-add-doc-description', type=str, default=None,
# #                       help='Whether to add doc description to the doc')
# #     parser.add_argument('--if-add-node-text', type=str, default=None,
# #                       help='Whether to add text to the node')
                      
# #     # Markdown specific arguments
# #     parser.add_argument('--if-thinning', type=str, default='no',
# #                       help='Whether to apply tree thinning for markdown (markdown only)')
# #     parser.add_argument('--thinning-threshold', type=int, default=5000,
# #                       help='Minimum token threshold for thinning (markdown only)')
# #     parser.add_argument('--summary-token-threshold', type=int, default=200,
# #                       help='Token threshold for generating summaries (markdown only)')
# #     args = parser.parse_args()
    
# #     # Validate that exactly one file type is specified
# #     if not args.pdf_path and not args.md_path:
# #         raise ValueError("Either --pdf_path or --md_path must be specified")
# #     if args.pdf_path and args.md_path:
# #         raise ValueError("Only one of --pdf_path or --md_path can be specified")
    
# #     if args.pdf_path:
# #         # Validate PDF file
# #         if not args.pdf_path.lower().endswith('.pdf'):
# #             raise ValueError("PDF file must have .pdf extension")
# #         if not os.path.isfile(args.pdf_path):
# #             raise ValueError(f"PDF file not found: {args.pdf_path}")
            
# #         # Process PDF file
# #         user_opt = {
# #             'model': args.model,
# #             'toc_check_page_num': args.toc_check_pages,
# #             'max_page_num_each_node': args.max_pages_per_node,
# #             'max_token_num_each_node': args.max_tokens_per_node,
# #             'if_add_node_id': args.if_add_node_id,
# #             'if_add_node_summary': args.if_add_node_summary,
# #             'if_add_doc_description': args.if_add_doc_description,
# #             'if_add_node_text': args.if_add_node_text,
# #         }
# #         opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

# #         # Process the PDF
# #         toc_with_page_number = page_index_main(args.pdf_path, opt)
# #         print('Parsing done, saving to file...')
        
# #         # Save results
# #         pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]    
# #         output_dir = './results'
# #         output_file = f'{output_dir}/{pdf_name}_structure.json'
# #         os.makedirs(output_dir, exist_ok=True)
        
# #         with open(output_file, 'w', encoding='utf-8') as f:
# #             json.dump(toc_with_page_number, f, indent=2)
        
# #         print(f'Tree structure saved to: {output_file}')
            
# #     elif args.md_path:
# #         # Validate Markdown file
# #         if not args.md_path.lower().endswith(('.md', '.markdown')):
# #             raise ValueError("Markdown file must have .md or .markdown extension")
# #         if not os.path.isfile(args.md_path):
# #             raise ValueError(f"Markdown file not found: {args.md_path}")
            
# #         # Process markdown file
# #         print('Processing markdown file...')
        
# #         # Process the markdown
# #         import asyncio
        
# #         # Use ConfigLoader to get consistent defaults (matching PDF behavior)
# #         from pageindex.utils import ConfigLoader
# #         config_loader = ConfigLoader()
        
# #         # Create options dict with user args
# #         user_opt = {
# #             'model': args.model,
# #             'if_add_node_summary': args.if_add_node_summary,
# #             'if_add_doc_description': args.if_add_doc_description,
# #             'if_add_node_text': args.if_add_node_text,
# #             'if_add_node_id': args.if_add_node_id
# #         }
        
# #         # Load config with defaults from config.yaml
# #         opt = config_loader.load(user_opt)
        
# #         toc_with_page_number = asyncio.run(md_to_tree(
# #             md_path=args.md_path,
# #             if_thinning=args.if_thinning.lower() == 'yes',
# #             min_token_threshold=args.thinning_threshold,
# #             if_add_node_summary=opt.if_add_node_summary,
# #             summary_token_threshold=args.summary_token_threshold,
# #             model=opt.model,
# #             if_add_doc_description=opt.if_add_doc_description,
# #             if_add_node_text=opt.if_add_node_text,
# #             if_add_node_id=opt.if_add_node_id
# #         ))
        
# #         print('Parsing done, saving to file...')
        
# #         # Save results
# #         md_name = os.path.splitext(os.path.basename(args.md_path))[0]    
# #         output_dir = './results'
# #         output_file = f'{output_dir}/{md_name}_structure.json'
# #         os.makedirs(output_dir, exist_ok=True)
        
# #         with open(output_file, 'w', encoding='utf-8') as f:
# #             json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
# #         print(f'Tree structure saved to: {output_file}')


# import argparse
# import os
# import json
# from pageindex import *
# from pageindex.page_index_md import md_to_tree
# from pageindex.utils import ConfigLoader

# if __name__ == "__main__":
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
#     parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
#     parser.add_argument('--pdf_dir', type=str, help='Path to a directory containing PDF files (batch mode)')
#     parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

#     parser.add_argument('--model', type=str, default=None, help='Model to use (overrides config.yaml)')

#     parser.add_argument('--toc-check-pages', type=int, default=None,
#                       help='Number of pages to check for table of contents (PDF only)')
#     parser.add_argument('--max-pages-per-node', type=int, default=None,
#                       help='Maximum number of pages per node (PDF only)')
#     parser.add_argument('--max-tokens-per-node', type=int, default=None,
#                       help='Maximum number of tokens per node (PDF only)')

#     parser.add_argument('--if-add-node-id', type=str, default=None,
#                       help='Whether to add node id to the node')
#     parser.add_argument('--if-add-node-summary', type=str, default=None,
#                       help='Whether to add summary to the node')
#     parser.add_argument('--if-add-doc-description', type=str, default=None,
#                       help='Whether to add doc description to the doc')
#     parser.add_argument('--if-add-node-text', type=str, default=None,
#                       help='Whether to add text to the node')

#     # Markdown specific arguments
#     parser.add_argument('--if-thinning', type=str, default='no',
#                       help='Whether to apply tree thinning for markdown (markdown only)')
#     parser.add_argument('--thinning-threshold', type=int, default=5000,
#                       help='Minimum token threshold for thinning (markdown only)')
#     parser.add_argument('--summary-token-threshold', type=int, default=200,
#                       help='Token threshold for generating summaries (markdown only)')
#     args = parser.parse_args()

#     # Validate that exactly one input type is specified
#     inputs = [args.pdf_path, args.pdf_dir, args.md_path]
#     if sum(x is not None for x in inputs) == 0:
#         raise ValueError("One of --pdf_path, --pdf_dir, or --md_path must be specified")
#     if sum(x is not None for x in inputs) > 1:
#         raise ValueError("Only one of --pdf_path, --pdf_dir, or --md_path can be specified at a time")

#     def process_single_pdf(pdf_path, opt):
#         """Process a single PDF and save its tree structure JSON."""
#         if not pdf_path.lower().endswith('.pdf'):
#             raise ValueError("PDF file must have .pdf extension")
#         if not os.path.isfile(pdf_path):
#             raise ValueError(f"PDF file not found: {pdf_path}")

#         toc_with_page_number = page_index_main(pdf_path, opt)
#         print('Parsing done, saving to file...')

#         pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
#         output_dir = './results'
#         output_file = f'{output_dir}/{pdf_name}_structure.json'
#         os.makedirs(output_dir, exist_ok=True)

#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(toc_with_page_number, f, indent=2)

#         print(f'Tree structure saved to: {output_file}')

#     if args.pdf_path or args.pdf_dir:
#         # Build shared options
#         user_opt = {
#             'model': args.model,
#             'toc_check_page_num': args.toc_check_pages,
#             'max_page_num_each_node': args.max_pages_per_node,
#             'max_token_num_each_node': args.max_tokens_per_node,
#             'if_add_node_id': args.if_add_node_id,
#             'if_add_node_summary': args.if_add_node_summary,
#             'if_add_doc_description': args.if_add_doc_description,
#             'if_add_node_text': args.if_add_node_text,
#         }
#         opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

#     if args.pdf_path:
#         process_single_pdf(args.pdf_path, opt)

#     elif args.pdf_dir:
#         if not os.path.isdir(args.pdf_dir):
#             raise ValueError(f"Directory not found: {args.pdf_dir}")

#         pdf_files = sorted([
#             f for f in os.listdir(args.pdf_dir) if f.lower().endswith('.pdf')
#         ])
#         if not pdf_files:
#             raise ValueError(f"No PDF files found in directory: {args.pdf_dir}")

#         print(f'Found {len(pdf_files)} PDF(s) in {args.pdf_dir}')
#         succeeded, failed = [], []

#         for i, filename in enumerate(pdf_files, 1):
#             pdf_path = os.path.join(args.pdf_dir, filename)
#             print(f'\n[{i}/{len(pdf_files)}] Processing: {filename}')
#             try:
#                 process_single_pdf(pdf_path, opt)
#                 succeeded.append(filename)
#             except Exception as e:
#                 print(f'Failed: {filename} — {e}')
#                 failed.append((filename, str(e)))

#         print(f'\nDone: {len(succeeded)}/{len(pdf_files)} succeeded')
#         if failed:
#             print(f'Failed ({len(failed)}):')
#             for name, err in failed:
#                 print(f'   - {name}: {err}')

#     elif args.md_path:
#         # Validate Markdown file
#         if not args.md_path.lower().endswith(('.md', '.markdown')):
#             raise ValueError("Markdown file must have .md or .markdown extension")
#         if not os.path.isfile(args.md_path):
#             raise ValueError(f"Markdown file not found: {args.md_path}")

#         print('Processing markdown file...')

#         import asyncio
#         from pageindex.utils import ConfigLoader
#         config_loader = ConfigLoader()

#         user_opt = {
#             'model': args.model,
#             'if_add_node_summary': args.if_add_node_summary,
#             'if_add_doc_description': args.if_add_doc_description,
#             'if_add_node_text': args.if_add_node_text,
#             'if_add_node_id': args.if_add_node_id
#         }

#         opt = config_loader.load(user_opt)

#         toc_with_page_number = asyncio.run(md_to_tree(
#             md_path=args.md_path,
#             if_thinning=args.if_thinning.lower() == 'yes',
#             min_token_threshold=args.thinning_threshold,
#             if_add_node_summary=opt.if_add_node_summary,
#             summary_token_threshold=args.summary_token_threshold,
#             model=opt.model,
#             if_add_doc_description=opt.if_add_doc_description,
#             if_add_node_text=opt.if_add_node_text,
#             if_add_node_id=opt.if_add_node_id
#         ))

#         print('Parsing done, saving to file...')

#         md_name = os.path.splitext(os.path.basename(args.md_path))[0]
#         output_dir = './results'
#         output_file = f'{output_dir}/{md_name}_structure.json'
#         os.makedirs(output_dir, exist_ok=True)

#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)

#         print(f'Tree structure saved to: {output_file}')


# import argparse
# import os
# import json
# from pageindex import *
# from pageindex.page_index_md import md_to_tree
# from pageindex.utils import ConfigLoader

# if __name__ == "__main__":
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
#     parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
#     parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

#     parser.add_argument('--model', type=str, default=None, help='Model to use (overrides config.yaml)')

#     parser.add_argument('--toc-check-pages', type=int, default=None,
#                       help='Number of pages to check for table of contents (PDF only)')
#     parser.add_argument('--max-pages-per-node', type=int, default=None,
#                       help='Maximum number of pages per node (PDF only)')
#     parser.add_argument('--max-tokens-per-node', type=int, default=None,
#                       help='Maximum number of tokens per node (PDF only)')

#     parser.add_argument('--if-add-node-id', type=str, default=None,
#                       help='Whether to add node id to the node')
#     parser.add_argument('--if-add-node-summary', type=str, default=None,
#                       help='Whether to add summary to the node')
#     parser.add_argument('--if-add-doc-description', type=str, default=None,
#                       help='Whether to add doc description to the doc')
#     parser.add_argument('--if-add-node-text', type=str, default=None,
#                       help='Whether to add text to the node')
                      
#     # Markdown specific arguments
#     parser.add_argument('--if-thinning', type=str, default='no',
#                       help='Whether to apply tree thinning for markdown (markdown only)')
#     parser.add_argument('--thinning-threshold', type=int, default=5000,
#                       help='Minimum token threshold for thinning (markdown only)')
#     parser.add_argument('--summary-token-threshold', type=int, default=200,
#                       help='Token threshold for generating summaries (markdown only)')
#     args = parser.parse_args()
    
#     # Validate that exactly one file type is specified
#     if not args.pdf_path and not args.md_path:
#         raise ValueError("Either --pdf_path or --md_path must be specified")
#     if args.pdf_path and args.md_path:
#         raise ValueError("Only one of --pdf_path or --md_path can be specified")
    
#     if args.pdf_path:
#         # Validate PDF file
#         if not args.pdf_path.lower().endswith('.pdf'):
#             raise ValueError("PDF file must have .pdf extension")
#         if not os.path.isfile(args.pdf_path):
#             raise ValueError(f"PDF file not found: {args.pdf_path}")
            
#         # Process PDF file
#         user_opt = {
#             'model': args.model,
#             'toc_check_page_num': args.toc_check_pages,
#             'max_page_num_each_node': args.max_pages_per_node,
#             'max_token_num_each_node': args.max_tokens_per_node,
#             'if_add_node_id': args.if_add_node_id,
#             'if_add_node_summary': args.if_add_node_summary,
#             'if_add_doc_description': args.if_add_doc_description,
#             'if_add_node_text': args.if_add_node_text,
#         }
#         opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

#         # Process the PDF
#         toc_with_page_number = page_index_main(args.pdf_path, opt)
#         print('Parsing done, saving to file...')
        
#         # Save results
#         pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]    
#         output_dir = './results'
#         output_file = f'{output_dir}/{pdf_name}_structure.json'
#         os.makedirs(output_dir, exist_ok=True)
        
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(toc_with_page_number, f, indent=2)
        
#         print(f'Tree structure saved to: {output_file}')
            
#     elif args.md_path:
#         # Validate Markdown file
#         if not args.md_path.lower().endswith(('.md', '.markdown')):
#             raise ValueError("Markdown file must have .md or .markdown extension")
#         if not os.path.isfile(args.md_path):
#             raise ValueError(f"Markdown file not found: {args.md_path}")
            
#         # Process markdown file
#         print('Processing markdown file...')
        
#         # Process the markdown
#         import asyncio
        
#         # Use ConfigLoader to get consistent defaults (matching PDF behavior)
#         from pageindex.utils import ConfigLoader
#         config_loader = ConfigLoader()
        
#         # Create options dict with user args
#         user_opt = {
#             'model': args.model,
#             'if_add_node_summary': args.if_add_node_summary,
#             'if_add_doc_description': args.if_add_doc_description,
#             'if_add_node_text': args.if_add_node_text,
#             'if_add_node_id': args.if_add_node_id
#         }
        
#         # Load config with defaults from config.yaml
#         opt = config_loader.load(user_opt)
        
#         toc_with_page_number = asyncio.run(md_to_tree(
#             md_path=args.md_path,
#             if_thinning=args.if_thinning.lower() == 'yes',
#             min_token_threshold=args.thinning_threshold,
#             if_add_node_summary=opt.if_add_node_summary,
#             summary_token_threshold=args.summary_token_threshold,
#             model=opt.model,
#             if_add_doc_description=opt.if_add_doc_description,
#             if_add_node_text=opt.if_add_node_text,
#             if_add_node_id=opt.if_add_node_id
#         ))
        
#         print('Parsing done, saving to file...')
        
#         # Save results
#         md_name = os.path.splitext(os.path.basename(args.md_path))[0]    
#         output_dir = './results'
#         output_file = f'{output_dir}/{md_name}_structure.json'
#         os.makedirs(output_dir, exist_ok=True)
        
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
#         print(f'Tree structure saved to: {output_file}')


import argparse
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pageindex import *
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--pdf_dir', type=str, help='Path to a directory containing PDF files (batch mode)')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

    parser.add_argument('--model', type=str, default=None, help='Model to use (overrides config.yaml)')

    parser.add_argument('--toc-check-pages', type=int, default=None,
                      help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--max-pages-per-node', type=int, default=None,
                      help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=None,
                      help='Maximum number of tokens per node (PDF only)')

    parser.add_argument('--if-add-node-id', type=str, default=None,
                      help='Whether to add node id to the node')
    parser.add_argument('--if-add-node-summary', type=str, default=None,
                      help='Whether to add summary to the node')
    parser.add_argument('--if-add-doc-description', type=str, default=None,
                      help='Whether to add doc description to the doc')
    parser.add_argument('--if-add-node-text', type=str, default=None,
                      help='Whether to add text to the node')
    parser.add_argument('--workers', type=int, default=1,
                      help='Number of PDFs to process in parallel for --pdf_dir (default: 1)')

    # Markdown specific arguments
    parser.add_argument('--if-thinning', type=str, default='no',
                      help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                      help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                      help='Token threshold for generating summaries (markdown only)')
    args = parser.parse_args()

    # Validate that exactly one input type is specified
    inputs = [args.pdf_path, args.pdf_dir, args.md_path]
    if sum(x is not None for x in inputs) == 0:
        raise ValueError("One of --pdf_path, --pdf_dir, or --md_path must be specified")
    if sum(x is not None for x in inputs) > 1:
        raise ValueError("Only one of --pdf_path, --pdf_dir, or --md_path can be specified at a time")

    def process_single_pdf(pdf_path, opt):
        """Process a single PDF and save its tree structure JSON."""
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(pdf_path):
            raise ValueError(f"PDF file not found: {pdf_path}")

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = './results'
        output_file = f'{output_dir}/{pdf_name}_structure.json'

        # Skip if already successfully processed — output file exists and is valid JSON
        if os.path.isfile(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    json.load(f)  # validates it is not corrupt/empty
                print(f'[SKIP] Already processed: {pdf_name}')
                return
            except (json.JSONDecodeError, ValueError):
                print(f'[REDO] Output file is corrupt, reprocessing: {pdf_name}')

        toc_with_page_number = page_index_main(pdf_path, opt)
        print('Parsing done, saving to file...')

        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2)

        print(f'Tree structure saved to: {output_file}')

    if args.pdf_path or args.pdf_dir:
        # Build shared options
        user_opt = {
            'model': args.model,
            'toc_check_page_num': args.toc_check_pages,
            'max_page_num_each_node': args.max_pages_per_node,
            'max_token_num_each_node': args.max_tokens_per_node,
            'if_add_node_id': args.if_add_node_id,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
        }
        opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

    if args.pdf_path:
        process_single_pdf(args.pdf_path, opt)

    elif args.pdf_dir:
        if not os.path.isdir(args.pdf_dir):
            raise ValueError(f"Directory not found: {args.pdf_dir}")

        pdf_files = sorted([
            f for f in os.listdir(args.pdf_dir) if f.lower().endswith('.pdf')
        ])
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {args.pdf_dir}")

        workers = max(1, args.workers)
        print(f'Found {len(pdf_files)} PDF(s) in {args.pdf_dir} (workers={workers})')
        succeeded, failed = [], []

        def _run(filename):
            pdf_path = os.path.join(args.pdf_dir, filename)
            process_single_pdf(pdf_path, opt)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run, f): f for f in pdf_files}
            done = 0
            for future in as_completed(futures):
                filename = futures[future]
                done += 1
                try:
                    future.result()
                    succeeded.append(filename)
                    print(f'[{done}/{len(pdf_files)}] Done: {filename}')
                except Exception as e:
                    failed.append((filename, str(e)))
                    print(f'[{done}/{len(pdf_files)}] Failed: {filename} — {e}')

        print(f'\nDone: {len(succeeded)}/{len(pdf_files)} succeeded')
        if failed:
            print(f'Failed ({len(failed)}):')
            for name, err in failed:
                print(f'   - {name}: {err}')

    elif args.md_path:
        # Validate Markdown file
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise ValueError(f"Markdown file not found: {args.md_path}")

        print('Processing markdown file...')

        import asyncio
        from pageindex.utils import ConfigLoader
        config_loader = ConfigLoader()

        user_opt = {
            'model': args.model,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
            'if_add_node_id': args.if_add_node_id
        }

        opt = config_loader.load(user_opt)

        toc_with_page_number = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id
        ))

        print('Parsing done, saving to file...')

        md_name = os.path.splitext(os.path.basename(args.md_path))[0]
        output_dir = './results'
        output_file = f'{output_dir}/{md_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)

        print(f'Tree structure saved to: {output_file}')