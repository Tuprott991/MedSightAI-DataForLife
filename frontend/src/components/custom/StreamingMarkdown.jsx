import React from 'react';

const urlPattern = /^https?:\/\//i;

const renderInline = (text, keyPrefix = 'inline') => {
    const nodes = [];
    let index = 0;

    const pushText = (value) => {
        if (value) nodes.push(value);
    };

    while (index < text.length) {
        if (text.startsWith('`', index)) {
            const end = text.indexOf('`', index + 1);
            if (end === -1) {
                pushText(text.slice(index));
                break;
            }
            nodes.push(
                <code key={`${keyPrefix}-code-${index}`} className="rounded bg-black/25 px-1 py-0.5 font-mono text-[0.92em]">
                    {text.slice(index + 1, end)}
                </code>
            );
            index = end + 1;
            continue;
        }

        if (text.startsWith('**', index)) {
            const end = text.indexOf('**', index + 2);
            if (end === -1) {
                pushText(text.slice(index));
                break;
            }
            nodes.push(
                <strong key={`${keyPrefix}-strong-${index}`} className="font-semibold">
                    {renderInline(text.slice(index + 2, end), `${keyPrefix}-strong-${index}`)}
                </strong>
            );
            index = end + 2;
            continue;
        }

        if (text.startsWith('*', index)) {
            const end = text.indexOf('*', index + 1);
            if (end === -1) {
                pushText(text.slice(index));
                break;
            }
            nodes.push(
                <em key={`${keyPrefix}-em-${index}`} className="italic">
                    {renderInline(text.slice(index + 1, end), `${keyPrefix}-em-${index}`)}
                </em>
            );
            index = end + 1;
            continue;
        }

        if (text.startsWith('[', index)) {
            const closeLabel = text.indexOf(']', index + 1);
            const openUrl = closeLabel !== -1 ? text.indexOf('(', closeLabel + 1) : -1;
            const closeUrl = openUrl !== -1 ? text.indexOf(')', openUrl + 1) : -1;
            if (closeLabel !== -1 && openUrl === closeLabel + 1 && closeUrl !== -1) {
                const label = text.slice(index + 1, closeLabel);
                const href = text.slice(openUrl + 1, closeUrl);
                if (urlPattern.test(href)) {
                    nodes.push(
                        <a
                            key={`${keyPrefix}-link-${index}`}
                            href={href}
                            target="_blank"
                            rel="noreferrer"
                            className="text-teal-300 underline decoration-teal-300/50 underline-offset-2"
                        >
                            {renderInline(label, `${keyPrefix}-link-${index}`)}
                        </a>
                    );
                    index = closeUrl + 1;
                    continue;
                }
            }
        }

        const nextSpecial = ['`', '*', '[']
            .map((char) => text.indexOf(char, index + 1))
            .filter((pos) => pos !== -1)
            .sort((a, b) => a - b)[0];
        const end = nextSpecial ?? text.length;
        pushText(text.slice(index, end));
        index = end;
    }

    return nodes;
};

const parseBlocks = (text) => {
    const lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
    const blocks = [];
    let index = 0;
    const splitTableRow = (row) => row
        .trim()
        .replace(/^\|/, '')
        .replace(/\|$/, '')
        .split('|')
        .map((cell) => cell.trim());
    const isTableDelimiter = (row) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(row);

    while (index < lines.length) {
        const line = lines[index];

        if (line.trim() === '') {
            blocks.push({ type: 'space' });
            index += 1;
            continue;
        }

        if (line.trimStart().startsWith('```')) {
            const codeLines = [];
            index += 1;
            while (index < lines.length && !lines[index].trimStart().startsWith('```')) {
                codeLines.push(lines[index]);
                index += 1;
            }
            if (index < lines.length) index += 1;
            blocks.push({ type: 'code', text: codeLines.join('\n') });
            continue;
        }

        const heading = /^(#{1,4})\s+(.+)$/.exec(line);
        if (heading) {
            blocks.push({ type: 'heading', level: heading[1].length, text: heading[2] });
            index += 1;
            continue;
        }

        if (/^\s*[-*]\s+/.test(line)) {
            const items = [];
            while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
                items.push(lines[index].replace(/^\s*[-*]\s+/, ''));
                index += 1;
            }
            blocks.push({ type: 'ul', items });
            continue;
        }

        if (/^\s*\d+[.)]\s+/.test(line)) {
            const items = [];
            while (index < lines.length && /^\s*\d+[.)]\s+/.test(lines[index])) {
                items.push(lines[index].replace(/^\s*\d+[.)]\s+/, ''));
                index += 1;
            }
            blocks.push({ type: 'ol', items });
            continue;
        }

        if (/^\s*>\s?/.test(line)) {
            const quoteLines = [];
            while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
                quoteLines.push(lines[index].replace(/^\s*>\s?/, ''));
                index += 1;
            }
            blocks.push({ type: 'quote', text: quoteLines.join('\n') });
            continue;
        }

        if (line.includes('|') && index + 1 < lines.length && isTableDelimiter(lines[index + 1])) {
            const headers = splitTableRow(line);
            const rows = [];
            index += 2;
            while (index < lines.length && lines[index].includes('|') && lines[index].trim() !== '') {
                rows.push(splitTableRow(lines[index]));
                index += 1;
            }
            blocks.push({ type: 'table', headers, rows });
            continue;
        }

        const paragraph = [line];
        index += 1;
        while (
            index < lines.length
            && lines[index].trim() !== ''
            && !lines[index].trimStart().startsWith('```')
            && !/^(#{1,4})\s+/.test(lines[index])
            && !/^\s*[-*]\s+/.test(lines[index])
            && !/^\s*\d+[.)]\s+/.test(lines[index])
            && !/^\s*>\s?/.test(lines[index])
        ) {
            paragraph.push(lines[index]);
            index += 1;
        }
        blocks.push({ type: 'paragraph', text: paragraph.join('\n') });
    }

    return blocks;
};

export const StreamingMarkdown = ({ text, isStreaming = false }) => {
    const blocks = parseBlocks(text);

    return (
        <div className="streaming-markdown space-y-2 leading-relaxed">
            {blocks.map((block, index) => {
                if (block.type === 'space') {
                    return <div key={index} className="h-1" />;
                }
                if (block.type === 'code') {
                    return (
                        <pre key={index} className="overflow-x-auto rounded bg-black/25 p-2 text-left font-mono text-xs">
                            <code>{block.text}</code>
                        </pre>
                    );
                }
                if (block.type === 'heading') {
                    const className = block.level <= 2 ? 'font-semibold text-white' : 'font-semibold';
                    return <p key={index} className={className}>{renderInline(block.text, `h-${index}`)}</p>;
                }
                if (block.type === 'ul') {
                    return (
                        <ul key={index} className="list-disc space-y-1 pl-5 text-left">
                            {block.items.map((item, itemIndex) => (
                                <li key={itemIndex}>{renderInline(item, `ul-${index}-${itemIndex}`)}</li>
                            ))}
                        </ul>
                    );
                }
                if (block.type === 'ol') {
                    return (
                        <ol key={index} className="list-decimal space-y-1 pl-5 text-left">
                            {block.items.map((item, itemIndex) => (
                                <li key={itemIndex}>{renderInline(item, `ol-${index}-${itemIndex}`)}</li>
                            ))}
                        </ol>
                    );
                }
                if (block.type === 'quote') {
                    return (
                        <blockquote key={index} className="border-l-2 border-teal-400/50 pl-3 text-left text-gray-300">
                            {renderInline(block.text, `quote-${index}`)}
                        </blockquote>
                    );
                }
                if (block.type === 'table') {
                    return (
                        <div key={index} className="overflow-x-auto">
                            <table className="min-w-full border-collapse text-left text-xs">
                                <thead>
                                    <tr>
                                        {block.headers.map((header, headerIndex) => (
                                            <th key={headerIndex} className="border border-white/15 px-2 py-1 font-semibold">
                                                {renderInline(header, `table-${index}-h-${headerIndex}`)}
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {block.rows.map((row, rowIndex) => (
                                        <tr key={rowIndex}>
                                            {block.headers.map((_, cellIndex) => (
                                                <td key={cellIndex} className="border border-white/10 px-2 py-1 align-top">
                                                    {renderInline(row[cellIndex] || '', `table-${index}-${rowIndex}-${cellIndex}`)}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    );
                }
                return (
                    <p key={index} className="whitespace-pre-wrap">
                        {renderInline(block.text, `p-${index}`)}
                    </p>
                );
            })}
            {isStreaming && <span className="inline-block animate-pulse">|</span>}
        </div>
    );
};
