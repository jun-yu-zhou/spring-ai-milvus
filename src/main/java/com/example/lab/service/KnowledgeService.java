package com.example.lab.service;

import org.springframework.ai.document.Document;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

/**
 * 知识库入库服务 —— RAG 中 ETL 流程的后两步（Transform 切分 + Load 装载）。
 *
 * <p>RAG 知识库的完整链路分三步，本项目这样分工：
 * <ol>
 *   <li><b>E（Extract 提取）</b>：由各类 DocumentReader 完成，例如
 *       PagePdfDocumentReader 逐页读 PDF、TextReader 读纯文本，
 *       产出 {@link Document}（文本内容 + 元数据）。</li>
 *   <li><b>T（Transform 切分）</b>：本服务用 {@link TokenTextSplitter}
 *       把长文档切成大小适中的片段（chunk）。</li>
 *   <li><b>L（Load 装载）</b>：本服务调用 {@link VectorStore#write} 写入 Milvus。
 *       写入时框架会自动调用配置的 Embedding 模型（text-embedding-v4）把每块文本
 *       转成 1024 维向量，连同元数据一起存进 vector_store 集合。</li>
 * </ol>
 *
 * <p>为什么要切分？——切分粒度直接决定检索质量：
 * <ul>
 *   <li>块太大：一块里混了多个主题，向量是"平均语义"，和具体问题都算不上相似；</li>
 *   <li>块太小：语义碎片化，模型即使检索到也凑不出完整上下文。</li>
 * </ul>
 */
@Service
public class KnowledgeService {

    private final VectorStore vectorStore;

    /**
     * 文本切分器：TokenTextSplitter 按 token 数切块，并尽量在"句子边界"断开。
     *
     * <p>它默认只在英文标点处断句，对中文文档会整段不切或切得很难看，
     * 所以这里必须把中文标点（。！？；）也加入分隔符列表。
     */
    private final TokenTextSplitter splitter = TokenTextSplitter.builder()
            .withChunkSize(700)            // 每块的目标长度（token 数），300~800 是常用区间
            .withMinChunkSizeChars(300)    // 块内最少字符数，太短的块会继续和相邻块合并
            .withMinChunkLengthToEmbed(20) // 少于该长度的块直接丢弃，避免噪音入库
            .withKeepSeparator(true)       // 保留分隔符本身（句号等），不丢失标点
            .withPunctuationMarks(List.of(
                    '。', '！', '？', '；', '\n',
                    '.', '!', '?', ';'
            ))
            .build();

    public KnowledgeService(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    /**
     * 通用入库入口：调用方负责构造 Document（如 PDF 读取器的产出，自带
     * file_name / page_number 等元数据），本服务负责切分与写入向量库。
     */
    public void ingest(List<Document> documents) {
        // splitter.apply()：切分，返回切好的 Document 列表（元数据自动继承自原文档）；
        // vectorStore.write()：装载，内部自动完成"向量化 + 写入 Milvus"两件事。
        vectorStore.write(splitter.apply(documents));
    }

    /**
     * 便捷入口：纯文本场景，手动补上元数据后复用通用入口。
     *
     * <p>元数据不是可有可无的：file_name 是后续"按文件删除旧知识"的依据
     * （向量库不会自动去重，同一文件重复入库会存两份，见 KnowledgeIngestTest）；
     * file_name / file_type 等字段以后还可以作为检索时的过滤条件。
     */
    public void ingest(String fileName, String fileType, String content) {
        ingest(List.of(new Document(content, Map.of(
                "file_name", fileName,
                "file_type", fileType,
                "type", "knowledge"
        ))));
    }
}
